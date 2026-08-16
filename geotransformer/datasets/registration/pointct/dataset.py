import json
import re
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from geotransformer.datasets.registration.pointct.coordinates import (
    normalize_axis_order,
    validate_rigid_transform,
)


POINT_TO_CT_DIRECTION = 'Point Cloud -> CT'
SITK_ARRAY_AXIS_ORDER = ('z', 'y', 'x')
IMAGE_INDEX_CONVENTION = ('x', 'y', 'z')
LPS_SPACE = 'left-posterior-superior'
DEFECT_ARTIFACT_FILENAMES = {
    'pointcloud': 'face_pointcloud_defect.npz',
    'ct': 'ct_defect.nrrd',
    'point_mask': 'Mp_gt.npy',
    'ct_mask': 'Mv_gt.nrrd',
    'metadata': 'metadata_defect.json',
}


class DatasetContractError(RuntimeError):
    pass


class NRRDDependencyError(RuntimeError):
    pass


def _read_json(path: Path) -> Dict:
    try:
        with path.open('r', encoding='utf-8') as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise DatasetContractError(f'Cannot read JSON metadata {path}: {error}') from error
    if not isinstance(value, dict):
        raise DatasetContractError(f'JSON metadata must be an object: {path}.')
    return value


def _read_nrrd_header(path: Path) -> Dict:
    fields = {}
    try:
        with path.open('rb') as handle:
            magic = handle.readline().decode('ascii').strip()
            if not magic.startswith('NRRD'):
                raise DatasetContractError(f'Invalid NRRD magic in {path}: {magic!r}.')
            for _ in range(256):
                raw_line = handle.readline()
                if raw_line in (b'', b'\n', b'\r\n'):
                    break
                line = raw_line.decode('ascii').strip()
                if not line or line.startswith('#'):
                    continue
                if ':' not in line:
                    raise DatasetContractError(f'Invalid NRRD header line in {path}: {line!r}.')
                key, value = line.split(':', 1)
                fields[key.strip().lower()] = value.strip()
            else:
                raise DatasetContractError(f'NRRD header is unexpectedly long: {path}.')
    except DatasetContractError:
        raise
    except (OSError, UnicodeDecodeError) as error:
        raise DatasetContractError(f'Cannot read NRRD header {path}: {error}') from error
    return fields


def _parse_nrrd_vector(value: str, field_name: str) -> np.ndarray:
    match = re.fullmatch(r'\(([^()]*)\)', value.strip())
    if match is None:
        raise DatasetContractError(f'Invalid NRRD {field_name}: {value!r}.')
    try:
        vector = np.asarray([float(part.strip()) for part in match.group(1).split(',')], dtype=np.float64)
    except ValueError as error:
        raise DatasetContractError(f'Invalid numeric NRRD {field_name}: {value!r}.') from error
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise DatasetContractError(f'NRRD {field_name} must be a finite 3-vector.')
    return vector


def _parse_nrrd_geometry(path: Path, array_axis_order) -> Dict:
    fields = _read_nrrd_header(path)
    required = {'type', 'dimension', 'space', 'sizes', 'space directions', 'space origin'}
    missing = required.difference(fields)
    if missing:
        raise DatasetContractError(f'NRRD header {path} is missing fields: {sorted(missing)}.')
    if fields['dimension'] != '3':
        raise DatasetContractError(f'NRRD must be 3D; got dimension={fields["dimension"]!r}.')
    try:
        image_size = np.asarray([int(value) for value in fields['sizes'].split()], dtype=np.int64)
    except ValueError as error:
        raise DatasetContractError(f'Invalid NRRD sizes in {path}: {fields["sizes"]!r}.') from error
    if image_size.shape != (3,) or np.any(image_size <= 0):
        raise DatasetContractError(f'NRRD sizes must contain three positive integers: {path}.')

    direction_tokens = re.findall(r'\([^()]*\)', fields['space directions'])
    if len(direction_tokens) != 3:
        raise DatasetContractError(f'NRRD space directions must contain three vectors: {path}.')
    space_directions = np.stack(
        [_parse_nrrd_vector(token, 'space directions') for token in direction_tokens], axis=1
    )
    spacing = np.linalg.norm(space_directions, axis=0)
    if np.any(spacing <= 0):
        raise DatasetContractError(f'NRRD space directions contain a zero-length axis: {path}.')
    direction = space_directions / spacing
    origin = _parse_nrrd_vector(fields['space origin'], 'space origin')

    image_size_by_axis = {'x': image_size[0], 'y': image_size[1], 'z': image_size[2]}
    axis_order = normalize_axis_order(array_axis_order)
    array_shape = np.asarray([image_size_by_axis[axis] for axis in axis_order], dtype=np.int64)
    type_map = {
        'char': 'int8',
        'signed char': 'int8',
        'uchar': 'uint8',
        'unsigned char': 'uint8',
        'short': 'int16',
        'short int': 'int16',
        'ushort': 'uint16',
        'unsigned short': 'uint16',
        'int': 'int32',
        'unsigned int': 'uint32',
        'float': 'float32',
        'double': 'float64',
    }
    nrrd_type = fields['type'].lower()
    if nrrd_type not in type_map:
        raise DatasetContractError(f'Unsupported NRRD scalar type {fields["type"]!r} in {path}.')
    return {
        'fields': fields,
        'image_size': image_size,
        'array_shape': array_shape,
        'dtype': type_map[nrrd_type],
        'spacing': spacing,
        'origin': origin,
        'direction': direction,
        'space': fields['space'].lower(),
    }


def _as_vector(value, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise DatasetContractError(f'{name} must be a finite vector with shape (3,); got {array}.')
    return array


def _as_direction(value, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (9,):
        array = array.reshape(3, 3)
    if array.shape != (3, 3) or not np.all(np.isfinite(array)):
        raise DatasetContractError(f'{name} must be finite with shape (3, 3); got {array.shape}.')
    if abs(np.linalg.det(array)) < 1e-12:
        raise DatasetContractError(f'{name} must be invertible.')
    return array


def _as_scalar_string(value, name: str) -> str:
    array = np.asarray(value)
    if array.shape != ():
        raise DatasetContractError(f'{name} must be a scalar string; got shape {array.shape}.')
    scalar = array.item()
    if not isinstance(scalar, str) or not scalar.strip():
        raise DatasetContractError(f'{name} must be a non-empty scalar string.')
    return scalar.strip()


def _require_mapping(value, name: str) -> Mapping:
    if not isinstance(value, dict):
        raise DatasetContractError(f'{name} must be a JSON object.')
    return value


def _as_integer_vector(value, name: str, length: int, *, positive: bool = False) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (length,) or not np.issubdtype(array.dtype, np.number):
        raise DatasetContractError(f'{name} must contain {length} numeric integers.')
    numeric = array.astype(np.float64)
    if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.floor(numeric)):
        raise DatasetContractError(f'{name} must contain {length} finite integers.')
    if positive and np.any(numeric <= 0):
        raise DatasetContractError(f'{name} must contain positive integers.')
    return numeric.astype(np.int64)


def _as_nonnegative_integer(value, name: str) -> int:
    array = np.asarray(value)
    if array.shape != () or not np.issubdtype(array.dtype, np.number):
        raise DatasetContractError(f'{name} must be a non-negative integer.')
    numeric = float(array.item())
    if not np.isfinite(numeric) or numeric < 0 or numeric != np.floor(numeric):
        raise DatasetContractError(f'{name} must be a non-negative integer.')
    return int(numeric)


def _validate_mask_convention(value, name: str):
    if not isinstance(value, str) or not value.strip():
        raise DatasetContractError(f'{name} must declare 0=defect and 1=intact semantics.')
    clauses = [clause.strip().lower().replace(' ', '') for clause in value.split(';')]
    zero_clause = next((clause for clause in clauses if clause.startswith('0=')), None)
    one_clause = next((clause for clause in clauses if clause.startswith('1=')), None)
    if zero_clause is None or 'defect' not in zero_clause or one_clause is None or 'intact' not in one_clause:
        raise DatasetContractError(
            f'{name} must preserve 0=defect and 1=intact semantics; got {value!r}.'
        )


def _validate_binary_uint8(array, name: str) -> np.ndarray:
    array = np.asarray(array)
    if array.dtype != np.dtype(np.uint8):
        raise DatasetContractError(f'{name} on-disk dtype must be uint8; got {array.dtype}.')
    if not np.all((array == 0) | (array == 1)):
        raise DatasetContractError(f'{name} must contain only binary values {{0,1}}.')
    return array


def _is_lps_coordinate_system(value) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    normalised = re.sub(r'\s+', ' ', value.strip().lower())
    if re.search(r'\bras\b', normalised) or 'right-anterior-superior' in normalised:
        return False
    if re.search(r'\bnot\s+(?:an?\s+)?lps\b', normalised) or 'not left-posterior-superior' in normalised:
        return False
    if 'non-lps' in normalised or 'non lps' in normalised:
        return False
    return bool(re.search(r'\blps\b', normalised) or 'left-posterior-superior' in normalised)


def _is_defect_point_coordinate_system(value) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    normalised = re.sub(r'\s+', ' ', value.strip().lower())
    if normalised == 'ct physical coordinates':
        return True
    return 'physical coordinate' in normalised and _is_lps_coordinate_system(normalised)


def read_ct_nrrd(path: Path) -> Dict:
    """Read one NRRD with SimpleITK and expose its physical metadata.

    SimpleITK's GetArrayFromImage contract yields a NumPy array in [z, y, x]
    storage order while image spacing/origin/direction use [x, y, z].
    """
    try:
        import SimpleITK as sitk
    except ModuleNotFoundError as error:
        raise NRRDDependencyError(
            'SimpleITK is required to read canonical ct.nrrd files. '
            'Install nothing automatically; provision the project dependency before real-data CT validation.'
        ) from error

    try:
        image = sitk.ReadImage(str(path))
        if image.GetDimension() != 3:
            raise DatasetContractError(f'CT image must be 3D; got dimension {image.GetDimension()} for {path}.')
        volume = sitk.GetArrayFromImage(image)
    except DatasetContractError:
        raise
    except Exception as error:
        raise DatasetContractError(f'Cannot read CT NRRD {path}: {error}') from error

    return {
        'ct_volume': volume,
        'ct_spacing': np.asarray(image.GetSpacing(), dtype=np.float64),
        'ct_origin': np.asarray(image.GetOrigin(), dtype=np.float64),
        'ct_direction': np.asarray(image.GetDirection(), dtype=np.float64).reshape(3, 3),
        'array_axis_order': SITK_ARRAY_AXIS_ORDER,
        'image_index_convention': IMAGE_INDEX_CONVENTION,
    }


def _normalise_transform_direction(value: str) -> str:
    return str(value).strip().lower().replace('→', '->').replace(' ', '')


def _resolve_gt_transform(pair_metadata: Mapping) -> np.ndarray:
    explicit_transform = pair_metadata.get('point_to_ct_transform')
    if explicit_transform is None:
        if pair_metadata.get('point_ct_alignment') != 'already_in_same_physical_coordinate_system':
            raise DatasetContractError(
                'Point-to-CT transform is unresolved: no explicit transform and metadata does not confirm a shared '
                'physical coordinate system.'
            )
        relationship = pair_metadata.get('point_to_ct_relationship')
        if not isinstance(relationship, str) or not relationship.strip():
            raise DatasetContractError('Identity Point-to-CT transform requires explicit relationship metadata.')
        return np.eye(4, dtype=np.float64)

    direction = pair_metadata.get('transform_direction')
    if direction is None or _normalise_transform_direction(direction) not in {
        'pointcloud->ct',
        'point->ct',
        'point_to_ct',
    }:
        raise DatasetContractError(
            f'Explicit transform must declare Point Cloud -> CT direction; got {direction!r}.'
        )
    try:
        return validate_rigid_transform(explicit_transform)
    except ValueError as error:
        raise DatasetContractError(f'Invalid Point-to-CT transform: {error}') from error


class PointCTDataset:
    """Manifest-driven M1 dataset for independent Point and CT branches."""

    def __init__(
        self,
        data_root,
        manifest_name: str = 'dataset_manifest.json',
        ct_reader: Optional[Callable[[Path], Dict]] = None,
        normal_tolerance: float = 1e-2,
        defect_variants: Optional[Sequence[Tuple[str, str]]] = None,
    ):
        self.data_root = Path(data_root).expanduser().resolve()
        if not self.data_root.is_dir():
            raise DatasetContractError(f'Dataset root is missing or inaccessible: {self.data_root}.')
        self.manifest_path = (self.data_root / manifest_name).resolve()
        self._ensure_inside_root(self.manifest_path)
        self.manifest = _read_json(self.manifest_path)
        self.ct_reader = read_ct_nrrd if ct_reader is None else ct_reader
        self.normal_tolerance = float(normal_tolerance)
        if not np.isfinite(self.normal_tolerance) or self.normal_tolerance <= 0:
            raise ValueError('normal_tolerance must be finite and positive.')

        subjects = self.manifest.get('subjects')
        if not isinstance(subjects, list):
            raise DatasetContractError('dataset_manifest.json must contain a subjects list.')
        self.subjects_total_in_manifest = len(subjects)
        ready_records = []
        self.skipped_records = []
        identities = set()
        for record in subjects:
            self._validate_manifest_record(record)
            identity = (record['subject_id'], record['case_id'], record['derived_sample_id'])
            if identity in identities:
                raise DatasetContractError(f'Duplicate manifest identity: {identity}.')
            identities.add(identity)
            if record['ready_for_baseline'] is True:
                ready_records.append(dict(record))
            else:
                self.skipped_records.append(dict(record))

        declared_total = self.manifest.get('num_subjects_total')
        declared_ready = self.manifest.get('num_subjects_ready')
        if declared_total is not None and int(declared_total) != self.subjects_total_in_manifest:
            raise DatasetContractError('Manifest num_subjects_total does not match subjects list length.')
        if declared_ready is not None and int(declared_ready) != len(ready_records):
            raise DatasetContractError('Manifest num_subjects_ready does not match ready_for_baseline entries.')
        if not isinstance(self.manifest.get('physical_unit'), str) or not self.manifest['physical_unit'].strip():
            raise DatasetContractError('Manifest must define a non-empty physical_unit.')

        self.defect_enabled = defect_variants is not None
        if self.defect_enabled:
            self.records = self._select_defect_records(ready_records, defect_variants)
        else:
            # B0 is deliberately the original M3 record path. Do not construct,
            # enumerate, or validate any defects/ directory in this branch.
            self.records = ready_records

        for record in self.records:
            for field in ('ct_path', 'pointcloud_path', 'pair_metadata_path', 'ct_metadata_path', 'qc_path'):
                path = self._record_path(record, field)
                if not path.is_file():
                    raise DatasetContractError(
                        f'Ready subject {record["subject_id"]} is missing required file {field}: {path}.'
                    )
            if self.defect_enabled:
                for name, path in self._defect_artifact_paths(record).items():
                    if not path.is_file():
                        raise DatasetContractError(
                            f'Defect variant {(record["subject_id"], record["defect_id"])} is missing '
                            f'core artifact {name}: {path}.'
                        )

    def _select_defect_records(self, ready_records, defect_variants) -> List[Dict]:
        if isinstance(defect_variants, (str, bytes)):
            raise DatasetContractError('defect_variants must be a sequence of (subject_id, defect_id) pairs.')
        try:
            selections = list(defect_variants)
        except TypeError as error:
            raise DatasetContractError(
                'defect_variants must be a sequence of (subject_id, defect_id) pairs.'
            ) from error
        if not selections:
            raise DatasetContractError('M4 defect mode requires at least one explicit (subject_id, defect_id).')

        records_by_subject = {}
        for record in ready_records:
            records_by_subject.setdefault(record['subject_id'], []).append(record)

        selected_records = []
        selected_identities = set()
        for selection in selections:
            if not isinstance(selection, (tuple, list)) or len(selection) != 2:
                raise DatasetContractError(
                    f'Each defect selection must be a (subject_id, defect_id) pair; got {selection!r}.'
                )
            subject_id, defect_id = selection
            for value, name in ((subject_id, 'subject_id'), (defect_id, 'defect_id')):
                if not isinstance(value, str) or not value.strip():
                    raise DatasetContractError(f'Defect selection {name} must be a non-empty string.')
                if value != value.strip():
                    raise DatasetContractError(f'Defect selection {name} may not contain outer whitespace.')
            if Path(defect_id).name != defect_id or defect_id in {'.', '..'} or '/' in defect_id or '\\' in defect_id:
                raise DatasetContractError(f'defect_id must be one safe directory component; got {defect_id!r}.')

            identity = (subject_id, defect_id)
            if identity in selected_identities:
                raise DatasetContractError(f'Duplicate explicit defect selection: {identity}.')
            selected_identities.add(identity)

            candidates = records_by_subject.get(subject_id, [])
            if not candidates:
                raise DatasetContractError(f'Explicit defect subject {subject_id!r} is not a ready manifest subject.')
            if len(candidates) != 1:
                raise DatasetContractError(
                    f'Explicit defect selection for subject {subject_id!r} maps to {len(candidates)} manifest '
                    'records; selection is not unique.'
                )
            record = dict(candidates[0])
            record['defect_id'] = defect_id
            selected_records.append(record)
        return selected_records

    def _defect_directory(self, record: Mapping) -> Path:
        if 'defect_id' not in record:
            raise DatasetContractError('Defect mode record is missing explicit defect_id.')
        subject_root = self._record_path(record, 'pair_metadata_path').parent
        defects_root = (subject_root / 'defects').resolve()
        self._ensure_inside_root(defects_root)
        variant_dir = (defects_root / record['defect_id']).resolve()
        self._ensure_inside_root(variant_dir)
        if variant_dir.parent != defects_root or variant_dir.name != record['defect_id']:
            raise DatasetContractError(f'Invalid defect variant directory for {record["defect_id"]!r}.')
        if not variant_dir.is_dir():
            raise DatasetContractError(
                f'Defect variant directory does not exist for {(record["subject_id"], record["defect_id"])}: '
                f'{variant_dir}.'
            )
        return variant_dir

    def _defect_artifact_paths(self, record: Mapping) -> Dict[str, Path]:
        variant_dir = self._defect_directory(record)
        return {name: variant_dir / filename for name, filename in DEFECT_ARTIFACT_FILENAMES.items()}

    def _ensure_inside_root(self, path: Path):
        try:
            path.relative_to(self.data_root)
        except ValueError as error:
            raise DatasetContractError(f'Manifest path escapes dataset root: {path}.') from error

    @staticmethod
    def _validate_manifest_record(record):
        if not isinstance(record, dict):
            raise DatasetContractError('Each manifest subject record must be an object.')
        required = (
            'subject_id',
            'case_id',
            'derived_sample_id',
            'ready_for_baseline',
            'ct_path',
            'pointcloud_path',
            'pair_metadata_path',
            'ct_metadata_path',
            'qc_path',
        )
        missing = [field for field in required if field not in record]
        if missing:
            raise DatasetContractError(f'Manifest subject record is missing fields: {missing}.')
        for field in ('subject_id', 'case_id', 'derived_sample_id'):
            if not isinstance(record[field], str) or not record[field].strip():
                raise DatasetContractError(f'Manifest {field} must be a non-empty string.')
        if not isinstance(record['ready_for_baseline'], bool):
            raise DatasetContractError('ready_for_baseline must be a JSON boolean.')

    def _record_path(self, record: Mapping, field: str) -> Path:
        value = record.get(field)
        if not isinstance(value, str) or not value.strip():
            raise DatasetContractError(f'Manifest field {field} must be a non-empty relative path.')
        path = (self.data_root / value).resolve()
        self._ensure_inside_root(path)
        return path

    def __len__(self):
        return len(self.records)

    def get_record(self, index: int) -> Dict:
        return dict(self.records[index])

    @property
    def ready_subject_ids(self) -> List[str]:
        return [record['subject_id'] for record in self.records]

    @property
    def skipped_subject_ids(self) -> List[str]:
        return [record['subject_id'] for record in self.skipped_records]

    def load_case_metadata(self, index: int, *, resolve_gt_transform: bool = True) -> Dict:
        record = self.records[index]
        pair_metadata = _read_json(self._record_path(record, 'pair_metadata_path'))
        ct_metadata = _read_json(self._record_path(record, 'ct_metadata_path'))
        qc_metadata = _read_json(self._record_path(record, 'qc_path'))

        for field in ('subject_id', 'case_id', 'derived_sample_id'):
            if pair_metadata.get(field) != record[field]:
                raise DatasetContractError(
                    f'{record["subject_id"]}: pair metadata {field} does not match manifest.'
                )
        if qc_metadata.get('subject_id') != record['subject_id']:
            raise DatasetContractError(f'{record["subject_id"]}: QC subject_id does not match manifest.')
        for field in (
            'processing_success',
            'ct_readable',
            'pointcloud_readable',
            'coordinate_contract_resolved',
            'point_ct_alignment_resolved',
            'ready_for_baseline',
        ):
            if qc_metadata.get(field) is not True:
                raise DatasetContractError(f'{record["subject_id"]}: QC field {field} is not true for a ready case.')

        canonical_ct_file = pair_metadata.get('canonical_ct_file')
        canonical_pointcloud_file = pair_metadata.get('canonical_pointcloud_file')
        if canonical_ct_file != self._record_path(record, 'ct_path').name:
            raise DatasetContractError(f'{record["subject_id"]}: canonical CT filename mismatch.')
        if canonical_pointcloud_file != self._record_path(record, 'pointcloud_path').name:
            raise DatasetContractError(f'{record["subject_id"]}: canonical point-cloud filename mismatch.')

        manifest_unit = self.manifest['physical_unit']
        if pair_metadata.get('physical_unit') != manifest_unit or ct_metadata.get('physical_unit') != manifest_unit:
            raise DatasetContractError(f'{record["subject_id"]}: physical_unit metadata is inconsistent.')

        pair_axis_order = normalize_axis_order(pair_metadata.get('ct_array_axis_order'))
        ct_axis_order = normalize_axis_order(ct_metadata.get('array_axis_order'))
        if pair_axis_order != ct_axis_order:
            raise DatasetContractError(f'{record["subject_id"]}: pair/CT array axis order mismatch.')
        image_convention = normalize_axis_order(ct_metadata.get('image_index_convention'))
        if image_convention != IMAGE_INDEX_CONVENTION:
            raise DatasetContractError(
                f'{record["subject_id"]}: image index convention must be [x,y,z], got {image_convention}.'
            )

        spacing = _as_vector(ct_metadata.get('spacing'), 'ct_metadata.spacing')
        if np.any(spacing <= 0):
            raise DatasetContractError(f'{record["subject_id"]}: CT spacing must be positive.')
        origin = _as_vector(ct_metadata.get('origin'), 'ct_metadata.origin')
        direction = _as_direction(ct_metadata.get('direction'), 'ct_metadata.direction')
        shape = np.asarray(ct_metadata.get('shape'))
        if shape.shape != (3,) or not np.all(shape == np.floor(shape)) or np.any(shape <= 0):
            raise DatasetContractError(f'{record["subject_id"]}: CT shape must contain three positive integers.')
        shape = shape.astype(np.int64)

        nrrd_header = _parse_nrrd_geometry(self._record_path(record, 'ct_path'), ct_axis_order)
        if not np.array_equal(nrrd_header['array_shape'], shape):
            raise DatasetContractError(f'{record["subject_id"]}: NRRD header/CT metadata shape mismatch.')
        if nrrd_header['dtype'] != str(ct_metadata.get('dtype')):
            raise DatasetContractError(f'{record["subject_id"]}: NRRD header/CT metadata dtype mismatch.')
        if not np.allclose(nrrd_header['spacing'], spacing):
            raise DatasetContractError(f'{record["subject_id"]}: NRRD header/CT metadata spacing mismatch.')
        if not np.allclose(nrrd_header['origin'], origin):
            raise DatasetContractError(f'{record["subject_id"]}: NRRD header/CT metadata origin mismatch.')
        if not np.allclose(nrrd_header['direction'], direction):
            raise DatasetContractError(f'{record["subject_id"]}: NRRD header/CT metadata direction mismatch.')
        if nrrd_header['space'] != str(ct_metadata.get('nrrd_space', '')).lower():
            raise DatasetContractError(f'{record["subject_id"]}: NRRD header/CT metadata space mismatch.')

        if not np.allclose(spacing, _as_vector(pair_metadata.get('ct_spacing'), 'pair.ct_spacing')):
            raise DatasetContractError(f'{record["subject_id"]}: pair/CT spacing mismatch.')
        if not np.allclose(origin, _as_vector(pair_metadata.get('ct_origin'), 'pair.ct_origin')):
            raise DatasetContractError(f'{record["subject_id"]}: pair/CT origin mismatch.')
        if not np.allclose(direction, _as_direction(pair_metadata.get('ct_direction'), 'pair.ct_direction')):
            raise DatasetContractError(f'{record["subject_id"]}: pair/CT direction mismatch.')
        if list(shape) != list(pair_metadata.get('ct_array_shape', [])):
            raise DatasetContractError(f'{record["subject_id"]}: pair/CT array shape mismatch.')

        coordinate_system = pair_metadata.get('coordinate_system')
        if not isinstance(coordinate_system, str) or not coordinate_system.strip():
            raise DatasetContractError(f'{record["subject_id"]}: coordinate_system is unresolved.')
        xyz_semantics = pair_metadata.get('point_xyz_coordinate_semantics')
        if not isinstance(xyz_semantics, str) or 'physical coordinate' not in xyz_semantics.lower():
            raise DatasetContractError(f'{record["subject_id"]}: point xyz physical semantics are unresolved.')
        gt_transform = _resolve_gt_transform(pair_metadata) if resolve_gt_transform else None
        return {
            'pair_metadata': pair_metadata,
            'ct_metadata': ct_metadata,
            'qc_metadata': qc_metadata,
            'ct_shape': shape,
            'ct_spacing': spacing,
            'ct_origin': origin,
            'ct_direction': direction,
            'array_axis_order': ct_axis_order,
            'image_index_convention': image_convention,
            'physical_unit': manifest_unit,
            'coordinate_system': coordinate_system,
            'nrrd_header': nrrd_header,
            'gt_transform': gt_transform,
            'gt_transform_direction': POINT_TO_CT_DIRECTION,
        }

    def load_pointcloud(self, index: int) -> Dict:
        record = self.records[index]
        metadata = self.load_case_metadata(index)
        pair_metadata = metadata['pair_metadata']
        path = self._record_path(record, 'pointcloud_path')
        try:
            with np.load(path, allow_pickle=False) as archive:
                keys = tuple(archive.files)
                xyz_key = pair_metadata.get('point_xyz_key')
                if not isinstance(xyz_key, str) or xyz_key not in archive.files:
                    raise DatasetContractError(
                        f'{record["subject_id"]}: declared point xyz key {xyz_key!r} is missing; actual keys={keys}.'
                    )
                point_xyz_phys = np.asarray(archive[xyz_key])
                normal_key = pair_metadata.get('point_normal_key')
                if normal_key is None:
                    point_normal = None
                elif not isinstance(normal_key, str) or normal_key not in archive.files:
                    raise DatasetContractError(
                        f'{record["subject_id"]}: declared normal key {normal_key!r} is missing; actual keys={keys}.'
                    )
                else:
                    point_normal = np.asarray(archive[normal_key])

                declared_keys = pair_metadata.get('pointcloud_npz_keys')
                if declared_keys is not None and not set(declared_keys).issubset(keys):
                    raise DatasetContractError(f'{record["subject_id"]}: NPZ keys disagree with pair metadata.')
                embedded_point_count = (
                    int(np.asarray(archive['point_count']).item()) if 'point_count' in archive.files else None
                )
                metadata_version = (
                    str(np.asarray(archive['metadata_version']).item())
                    if 'metadata_version' in archive.files
                    else None
                )
                sampling_method = (
                    str(np.asarray(archive['sampling_method']).item())
                    if 'sampling_method' in archive.files
                    else None
                )
        except DatasetContractError:
            raise
        except Exception as error:
            raise DatasetContractError(f'Cannot read point cloud NPZ {path}: {error}') from error

        if point_xyz_phys.ndim != 2 or point_xyz_phys.shape[1] != 3 or point_xyz_phys.shape[0] == 0:
            raise DatasetContractError(
                f'{record["subject_id"]}: point xyz must have shape (N,3), N>0; got {point_xyz_phys.shape}.'
            )
        if not np.issubdtype(point_xyz_phys.dtype, np.number) or not np.all(np.isfinite(point_xyz_phys)):
            raise DatasetContractError(f'{record["subject_id"]}: point xyz must be finite numeric data.')
        point_count = int(point_xyz_phys.shape[0])
        for name, value in (
            ('NPZ point_count', embedded_point_count),
            ('pair point_count', pair_metadata.get('point_count')),
        ):
            if value is not None and int(value) != point_count:
                raise DatasetContractError(f'{record["subject_id"]}: {name} does not match xyz length.')

        if point_normal is not None:
            if point_normal.shape != point_xyz_phys.shape:
                raise DatasetContractError(
                    f'{record["subject_id"]}: normal shape {point_normal.shape} does not match xyz shape.'
                )
            if not np.issubdtype(point_normal.dtype, np.number) or not np.all(np.isfinite(point_normal)):
                raise DatasetContractError(f'{record["subject_id"]}: point normals must be finite numeric data.')
            normal_norm = np.linalg.norm(point_normal, axis=1)
            if not np.all(np.abs(normal_norm - 1.0) <= self.normal_tolerance):
                raise DatasetContractError(
                    f'{record["subject_id"]}: normal norms exceed tolerance {self.normal_tolerance}.'
                )

        return {
            'point_xyz_phys': point_xyz_phys,
            'point_normal': point_normal,
            'point_count': point_count,
            'pointcloud_npz_keys': keys,
            'metadata_version': metadata_version,
            'sampling_method': sampling_method,
        }

    def load_ct(self, index: int) -> Dict:
        record = self.records[index]
        metadata = self.load_case_metadata(index)
        path = self._record_path(record, 'ct_path')
        ct = self.ct_reader(path)
        required = {
            'ct_volume',
            'ct_spacing',
            'ct_origin',
            'ct_direction',
            'array_axis_order',
            'image_index_convention',
        }
        missing = required.difference(ct)
        if missing:
            raise DatasetContractError(f'{record["subject_id"]}: CT reader output is missing {sorted(missing)}.')

        volume = np.asarray(ct['ct_volume'])
        if volume.ndim != 3 or any(size <= 0 for size in volume.shape):
            raise DatasetContractError(f'{record["subject_id"]}: CT volume must be a non-empty 3D array.')
        if not np.issubdtype(volume.dtype, np.number):
            raise DatasetContractError(f'{record["subject_id"]}: CT volume dtype must be numeric.')
        if np.issubdtype(volume.dtype, np.floating) and not np.all(np.isfinite(volume)):
            raise DatasetContractError(f'{record["subject_id"]}: floating-point CT volume contains NaN or Inf.')
        if tuple(volume.shape) != tuple(metadata['ct_shape']):
            raise DatasetContractError(
                f'{record["subject_id"]}: CT array shape {volume.shape} disagrees with metadata '
                f'{tuple(metadata["ct_shape"])}.'
            )
        expected_dtype = str(metadata['ct_metadata'].get('dtype'))
        if str(volume.dtype) != expected_dtype:
            raise DatasetContractError(
                f'{record["subject_id"]}: CT dtype {volume.dtype} disagrees with metadata {expected_dtype}.'
            )
        reader_axis_order = normalize_axis_order(ct['array_axis_order'])
        reader_index_convention = normalize_axis_order(ct['image_index_convention'])
        if reader_axis_order != metadata['array_axis_order']:
            raise DatasetContractError(f'{record["subject_id"]}: CT reader/metadata array axis mismatch.')
        if reader_index_convention != metadata['image_index_convention']:
            raise DatasetContractError(f'{record["subject_id"]}: CT reader/metadata image index mismatch.')

        spacing = _as_vector(ct['ct_spacing'], 'reader.ct_spacing')
        origin = _as_vector(ct['ct_origin'], 'reader.ct_origin')
        direction = _as_direction(ct['ct_direction'], 'reader.ct_direction')
        if not np.allclose(spacing, metadata['ct_spacing']):
            raise DatasetContractError(f'{record["subject_id"]}: CT reader/metadata spacing mismatch.')
        if not np.allclose(origin, metadata['ct_origin']):
            raise DatasetContractError(f'{record["subject_id"]}: CT reader/metadata origin mismatch.')
        if not np.allclose(direction, metadata['ct_direction']):
            raise DatasetContractError(f'{record["subject_id"]}: CT reader/metadata direction mismatch.')

        return {
            'ct_volume': volume,
            'ct_spacing': spacing,
            'ct_origin': origin,
            'ct_direction': direction,
            'ct_shape': np.asarray(volume.shape, dtype=np.int64),
            'array_axis_order': reader_axis_order,
            'image_index_convention': reader_index_convention,
        }

    def load_defect_metadata(self, index: int) -> Dict:
        record = self.records[index]
        paths = self._defect_artifact_paths(record)
        metadata = _read_json(paths['metadata'])

        if metadata.get('patient_id') != record['subject_id']:
            raise DatasetContractError(
                f'{record["subject_id"]}: defect metadata patient_id does not match selected subject.'
            )
        if metadata.get('defect_id') != record['defect_id']:
            raise DatasetContractError(
                f'{record["subject_id"]}: defect metadata defect_id does not match explicit selection.'
            )

        ct_geometry = _require_mapping(metadata.get('ct_geometry'), 'metadata_defect.ct_geometry')
        mp_metadata = _require_mapping(metadata.get('Mp'), 'metadata_defect.Mp')
        mv_metadata = _require_mapping(metadata.get('Mv'), 'metadata_defect.Mv')
        _validate_mask_convention(mp_metadata.get('convention'), 'metadata_defect.Mp.convention')
        _validate_mask_convention(mv_metadata.get('convention'), 'metadata_defect.Mv.convention')

        formal_transform = metadata.get('formal_transform')
        if formal_transform != 'Identity':
            if formal_transform is None:
                raise DatasetContractError('metadata_defect.formal_transform is required in M4 defect mode.')
            raise DatasetContractError(
                f'Unsupported metadata_defect.formal_transform {formal_transform!r}; only "Identity" is frozen.'
            )
        try:
            gt_transform = validate_rigid_transform(np.eye(4, dtype=np.float64))
        except ValueError as error:
            raise DatasetContractError(f'Invalid defect Identity transform: {error}') from error

        return {
            'metadata': metadata,
            'ct_geometry': ct_geometry,
            'Mp': mp_metadata,
            'Mv': mv_metadata,
            'gt_transform': gt_transform,
            'gt_transform_direction': POINT_TO_CT_DIRECTION,
        }

    def load_defect_pointcloud(self, index: int, defect_metadata: Mapping) -> Dict:
        record = self.records[index]
        path = self._defect_artifact_paths(record)['pointcloud']
        try:
            with np.load(path, allow_pickle=False) as archive:
                keys = tuple(archive.files)
                missing = {'xyz', 'normal', 'defect_id'}.difference(keys)
                if missing:
                    raise DatasetContractError(
                        f'{record["subject_id"]}: defective point NPZ is missing keys {sorted(missing)}.'
                    )
                point_xyz_phys = np.asarray(archive['xyz'])
                point_normal = np.asarray(archive['normal'])
                embedded_defect_id = _as_scalar_string(archive['defect_id'], 'pointcloud.defect_id')
                optional_strings = {
                    name: _as_scalar_string(archive[name], f'pointcloud.{name}')
                    for name in ('patient_id', 'coordinate_system', 'unit', 'sampling_method')
                    if name in archive.files
                }
        except DatasetContractError:
            raise
        except Exception as error:
            raise DatasetContractError(f'Cannot read defective point cloud NPZ {path}: {error}') from error

        if embedded_defect_id != record['defect_id']:
            raise DatasetContractError(
                f'{record["subject_id"]}: NPZ defect_id {embedded_defect_id!r} does not match explicit selection '
                f'{record["defect_id"]!r}.'
            )
        patient_id = optional_strings.get('patient_id')
        if patient_id is not None and patient_id != record['subject_id']:
            raise DatasetContractError(
                f'{record["subject_id"]}: NPZ patient_id {patient_id!r} does not match selected subject.'
            )
        unit = optional_strings.get('unit')
        if unit is not None and unit != self.manifest['physical_unit']:
            raise DatasetContractError(
                f'{record["subject_id"]}: NPZ unit {unit!r} does not match dataset physical unit.'
            )
        coordinate_system = optional_strings.get('coordinate_system')
        if coordinate_system is not None and not _is_defect_point_coordinate_system(coordinate_system):
            raise DatasetContractError(
                f'{record["subject_id"]}: NPZ coordinate_system is not the accepted CT/LPS physical frame: '
                f'{coordinate_system!r}.'
            )
        sampling_method = optional_strings.get('sampling_method')
        declared_sampling_method = defect_metadata['metadata'].get('sampling_method')
        if sampling_method is not None and declared_sampling_method is not None and sampling_method != declared_sampling_method:
            raise DatasetContractError(
                f'{record["subject_id"]}: NPZ sampling_method disagrees with defect metadata.'
            )

        if point_xyz_phys.ndim != 2 or point_xyz_phys.shape[1] != 3 or point_xyz_phys.shape[0] == 0:
            raise DatasetContractError(
                f'{record["subject_id"]}: defective xyz must have shape (N,3), N>0; got {point_xyz_phys.shape}.'
            )
        if not np.issubdtype(point_xyz_phys.dtype, np.floating) or not np.all(np.isfinite(point_xyz_phys)):
            raise DatasetContractError(
                f'{record["subject_id"]}: defective xyz must contain finite floating-point values.'
            )
        if point_normal.shape != point_xyz_phys.shape:
            raise DatasetContractError(
                f'{record["subject_id"]}: defective normal shape {point_normal.shape} does not match xyz shape.'
            )
        if not np.issubdtype(point_normal.dtype, np.floating) or not np.all(np.isfinite(point_normal)):
            raise DatasetContractError(
                f'{record["subject_id"]}: defective normals must contain finite floating-point values.'
            )
        normal_norm = np.linalg.norm(point_normal, axis=1)
        if not np.all(np.abs(normal_norm - 1.0) <= self.normal_tolerance):
            raise DatasetContractError(
                f'{record["subject_id"]}: defective normal norms exceed tolerance {self.normal_tolerance}.'
            )

        point_count = int(point_xyz_phys.shape[0])
        declared_shape = _as_integer_vector(
            defect_metadata['Mp'].get('shape'), 'metadata_defect.Mp.shape', 1, positive=True
        )
        if declared_shape[0] != point_count:
            raise DatasetContractError(
                f'{record["subject_id"]}: metadata_defect.Mp.shape does not match defective point count.'
            )
        point_provenance = defect_metadata['metadata'].get('defect_pointcloud')
        if point_provenance is not None:
            point_provenance = _require_mapping(point_provenance, 'metadata_defect.defect_pointcloud')
            declared_count = _as_nonnegative_integer(
                point_provenance.get('point_count'), 'metadata_defect.defect_pointcloud.point_count'
            )
            if declared_count != point_count:
                raise DatasetContractError(
                    f'{record["subject_id"]}: defect point provenance point_count mismatch.'
                )

        return {
            'point_xyz_phys': point_xyz_phys,
            'point_normal': point_normal,
            'point_count': point_count,
            'pointcloud_npz_keys': keys,
            'sampling_method': sampling_method,
        }

    def load_defect_point_mask(self, index: int, point_count: int, defect_metadata: Mapping) -> np.ndarray:
        record = self.records[index]
        path = self._defect_artifact_paths(record)['point_mask']
        try:
            point_mask_uint8 = np.load(path, allow_pickle=False)
        except Exception as error:
            raise DatasetContractError(f'Cannot read Point defect mask {path}: {error}') from error
        point_mask_uint8 = _validate_binary_uint8(point_mask_uint8, 'Mp_gt')
        if point_mask_uint8.ndim != 1:
            raise DatasetContractError(f'Mp_gt must have shape [N]; got {point_mask_uint8.shape}.')
        if point_mask_uint8.shape[0] != point_count:
            raise DatasetContractError(
                f'Mp_gt length {point_mask_uint8.shape[0]} does not match defective point count {point_count}.'
            )

        mp_metadata = defect_metadata['Mp']
        zero_count = int(np.count_nonzero(point_mask_uint8 == 0))
        one_count = int(np.count_nonzero(point_mask_uint8 == 1))
        declared_zero_count = _as_nonnegative_integer(
            mp_metadata.get('zero_count'), 'metadata_defect.Mp.zero_count'
        )
        declared_one_count = _as_nonnegative_integer(
            mp_metadata.get('one_count'), 'metadata_defect.Mp.one_count'
        )
        if declared_zero_count != zero_count or declared_one_count != one_count:
            raise DatasetContractError('metadata_defect.Mp counts do not match Mp_gt.npy.')
        # Length agreement is necessary but does not prove kth-row identity;
        # that guarantee belongs to accepted generator provenance.
        return point_mask_uint8.astype(bool)

    def _read_defect_image(self, path: Path, name: str) -> Dict:
        header = _parse_nrrd_geometry(path, SITK_ARRAY_AXIS_ORDER)
        if header['space'] != LPS_SPACE:
            raise DatasetContractError(f'{name} NRRD space must be {LPS_SPACE!r}; got {header["space"]!r}.')
        image = self.ct_reader(path)
        required = {
            'ct_volume',
            'ct_spacing',
            'ct_origin',
            'ct_direction',
            'array_axis_order',
            'image_index_convention',
        }
        missing = required.difference(image)
        if missing:
            raise DatasetContractError(f'{name} reader output is missing {sorted(missing)}.')

        volume = np.asarray(image['ct_volume'])
        if volume.ndim != 3 or any(size <= 0 for size in volume.shape):
            raise DatasetContractError(f'{name} volume must be a non-empty [Z,Y,X] array.')
        if not np.issubdtype(volume.dtype, np.number):
            raise DatasetContractError(f'{name} volume must contain numeric values.')
        if np.issubdtype(volume.dtype, np.floating) and not np.all(np.isfinite(volume)):
            raise DatasetContractError(f'{name} volume contains NaN or Inf.')
        if tuple(volume.shape) != tuple(header['array_shape']):
            raise DatasetContractError(f'{name} reader/header shape mismatch.')
        if str(volume.dtype) != header['dtype']:
            raise DatasetContractError(
                f'{name} reader/header dtype mismatch: reader={volume.dtype}, header={header["dtype"]}.'
            )

        array_axis_order = normalize_axis_order(image['array_axis_order'])
        image_index_convention = normalize_axis_order(image['image_index_convention'])
        if array_axis_order != SITK_ARRAY_AXIS_ORDER:
            raise DatasetContractError(f'{name} array axis order must be [z,y,x].')
        if image_index_convention != IMAGE_INDEX_CONVENTION:
            raise DatasetContractError(f'{name} image index convention must be [x,y,z].')
        spacing = _as_vector(image['ct_spacing'], f'{name}.spacing')
        origin = _as_vector(image['ct_origin'], f'{name}.origin')
        direction = _as_direction(image['ct_direction'], f'{name}.direction')
        if np.any(spacing <= 0):
            raise DatasetContractError(f'{name} spacing must be positive.')
        if not np.array_equal(spacing, header['spacing']):
            raise DatasetContractError(f'{name} reader/header spacing mismatch.')
        if not np.array_equal(origin, header['origin']):
            raise DatasetContractError(f'{name} reader/header origin mismatch.')
        if not np.array_equal(direction, header['direction']):
            raise DatasetContractError(f'{name} reader/header direction mismatch.')

        return {
            'volume': volume,
            'spacing': spacing,
            'origin': origin,
            'direction': direction,
            'array_axis_order': array_axis_order,
            'image_index_convention': image_index_convention,
            'header': header,
        }

    def load_defect_ct(self, index: int, defect_metadata: Mapping) -> Dict:
        record = self.records[index]
        paths = self._defect_artifact_paths(record)
        ct = self._read_defect_image(paths['ct'], 'ct_defect')
        mv = self._read_defect_image(paths['ct_mask'], 'Mv_gt')

        ct_mask_uint8 = _validate_binary_uint8(mv['volume'], 'Mv_gt')
        if mv['header']['dtype'] != 'uint8':
            raise DatasetContractError(f'Mv_gt on-disk dtype must be uint8; got {mv["header"]["dtype"]}.')
        if ct['volume'].shape != ct_mask_uint8.shape:
            raise DatasetContractError('ct_defect/Mv_gt array shape mismatch.')
        for field in ('spacing', 'origin', 'direction'):
            if not np.array_equal(ct[field], mv[field]):
                raise DatasetContractError(f'ct_defect/Mv_gt {field} mismatch.')
        if ct['header']['space'] != mv['header']['space']:
            raise DatasetContractError('ct_defect/Mv_gt coordinate system mismatch.')

        ct_geometry = defect_metadata['ct_geometry']
        size_xyz = _as_integer_vector(
            ct_geometry.get('size_xyz'), 'metadata_defect.ct_geometry.size_xyz', 3, positive=True
        )
        if not np.array_equal(size_xyz, ct['header']['image_size']):
            raise DatasetContractError('metadata_defect.ct_geometry.size_xyz mismatch.')
        metadata_spacing = _as_vector(ct_geometry.get('spacing_xyz_mm'), 'metadata_defect.ct_geometry.spacing_xyz_mm')
        metadata_origin = _as_vector(ct_geometry.get('origin_xyz_mm'), 'metadata_defect.ct_geometry.origin_xyz_mm')
        metadata_direction = _as_direction(ct_geometry.get('direction'), 'metadata_defect.ct_geometry.direction')
        for name, actual, declared in (
            ('spacing', ct['spacing'], metadata_spacing),
            ('origin', ct['origin'], metadata_origin),
            ('direction', ct['direction'], metadata_direction),
        ):
            if not np.array_equal(actual, declared):
                raise DatasetContractError(f'metadata_defect CT {name} mismatch.')

        mv_metadata = defect_metadata['Mv']
        mv_shape = _as_integer_vector(
            mv_metadata.get('shape_zyx'), 'metadata_defect.Mv.shape_zyx', 3, positive=True
        )
        if not np.array_equal(mv_shape, np.asarray(ct_mask_uint8.shape)):
            raise DatasetContractError('metadata_defect.Mv.shape_zyx mismatch.')
        zero_count = int(np.count_nonzero(ct_mask_uint8 == 0))
        one_count = int(np.count_nonzero(ct_mask_uint8 == 1))
        declared_zero_count = _as_nonnegative_integer(
            mv_metadata.get('zero_count'), 'metadata_defect.Mv.zero_count'
        )
        declared_one_count = _as_nonnegative_integer(
            mv_metadata.get('one_count'), 'metadata_defect.Mv.one_count'
        )
        if declared_zero_count != zero_count or declared_one_count != one_count:
            raise DatasetContractError('metadata_defect.Mv counts do not match Mv_gt.nrrd.')

        return {
            'ct_volume': ct['volume'],
            'ct_spacing': ct['spacing'],
            'ct_origin': ct['origin'],
            'ct_direction': ct['direction'],
            'ct_shape': np.asarray(ct['volume'].shape, dtype=np.int64),
            'array_axis_order': ct['array_axis_order'],
            'image_index_convention': ct['image_index_convention'],
            'ct_defect_mask': ct_mask_uint8.astype(bool),
        }

    def __getitem__(self, index: int) -> Dict:
        if self.defect_enabled:
            return self._get_defect_item(index)

        record = self.records[index]
        metadata = self.load_case_metadata(index)
        point = self.load_pointcloud(index)
        ct = self.load_ct(index)
        return {
            'subject_id': record['subject_id'],
            'case_id': record['case_id'],
            'derived_sample_id': record['derived_sample_id'],
            'point_xyz_phys': point['point_xyz_phys'],
            'point_normal': point['point_normal'],
            'ct_volume': ct['ct_volume'],
            'ct_spacing': ct['ct_spacing'],
            'ct_origin': ct['ct_origin'],
            'ct_direction': ct['ct_direction'],
            'ct_shape': ct['ct_shape'],
            'ct_array_axis_order': ct['array_axis_order'],
            'ct_image_index_convention': ct['image_index_convention'],
            'gt_transform': metadata['gt_transform'],
            'gt_transform_direction': metadata['gt_transform_direction'],
            'physical_unit': metadata['physical_unit'],
            'coordinate_system': metadata['coordinate_system'],
            'point_count': point['point_count'],
            'pair_metadata': metadata['pair_metadata'],
            'ct_metadata': metadata['ct_metadata'],
        }

    def _get_defect_item(self, index: int) -> Dict:
        record = self.records[index]
        # Keep the existing M3 identity/metadata contract, but deliberately do
        # not resolve or use the parent pair's transform for a defect sample.
        parent_metadata = self.load_case_metadata(index, resolve_gt_transform=False)
        if parent_metadata['physical_unit'] != 'mm':
            raise DatasetContractError(
                f'{record["subject_id"]}: M4 defect physical unit must be mm.'
            )
        if not _is_lps_coordinate_system(parent_metadata['coordinate_system']):
            raise DatasetContractError(
                f'{record["subject_id"]}: M4 defect coordinate system must be physical LPS.'
            )

        defect_metadata = self.load_defect_metadata(index)
        point = self.load_defect_pointcloud(index, defect_metadata)
        point_defect_mask = self.load_defect_point_mask(index, point['point_count'], defect_metadata)
        ct = self.load_defect_ct(index, defect_metadata)

        return {
            'subject_id': record['subject_id'],
            'case_id': record['case_id'],
            'derived_sample_id': record['derived_sample_id'],
            'defect_id': record['defect_id'],
            'point_xyz_phys': point['point_xyz_phys'],
            'point_normal': point['point_normal'],
            'point_defect_mask': point_defect_mask,
            'ct_volume': ct['ct_volume'],
            'ct_spacing': ct['ct_spacing'],
            'ct_origin': ct['ct_origin'],
            'ct_direction': ct['ct_direction'],
            'ct_shape': ct['ct_shape'],
            'ct_array_axis_order': ct['array_axis_order'],
            'ct_image_index_convention': ct['image_index_convention'],
            'ct_defect_mask': ct['ct_defect_mask'],
            'gt_transform': defect_metadata['gt_transform'],
            'gt_transform_direction': defect_metadata['gt_transform_direction'],
            'physical_unit': parent_metadata['physical_unit'],
            'coordinate_system': parent_metadata['coordinate_system'],
            'point_count': point['point_count'],
            'pair_metadata': parent_metadata['pair_metadata'],
            'ct_metadata': parent_metadata['ct_metadata'],
        }


def pointct_collate_fn(samples: List[Mapping]) -> Dict:
    """Keep Point and CT as independent branches; M1 supports batch_size=1 only."""
    if len(samples) != 1:
        raise DatasetContractError(f'M1 Point-CT collate requires batch_size=1; got {len(samples)} samples.')
    sample = samples[0]
    collated = {
        'subject_id': sample['subject_id'],
        'case_id': sample['case_id'],
        'derived_sample_id': sample['derived_sample_id'],
        'point': {
            'point_xyz_phys': sample['point_xyz_phys'],
            'point_normal': sample['point_normal'],
            'point_count': sample['point_count'],
        },
        'ct': {
            'ct_volume': sample['ct_volume'],
            'ct_spacing': sample['ct_spacing'],
            'ct_origin': sample['ct_origin'],
            'ct_direction': sample['ct_direction'],
            'ct_shape': sample['ct_shape'],
            'array_axis_order': sample['ct_array_axis_order'],
            'image_index_convention': sample['ct_image_index_convention'],
        },
        'gt_transform': sample['gt_transform'],
        'gt_transform_direction': sample['gt_transform_direction'],
        'physical_unit': sample['physical_unit'],
        'coordinate_system': sample['coordinate_system'],
    }
    if 'defect_id' in sample:
        if 'point_defect_mask' not in sample or 'ct_defect_mask' not in sample:
            raise DatasetContractError('Defect sample must provide both raw Point and CT defect masks.')
        collated['defect_id'] = sample['defect_id']
        collated['point']['point_defect_mask'] = sample['point_defect_mask']
        collated['ct']['ct_defect_mask'] = sample['ct_defect_mask']
    return collated


def find_subject_split_leakage(records: Iterable[Mapping]) -> Dict[str, List[str]]:
    subject_splits: Dict[str, set] = {}
    for record in records:
        subject_id = record.get('subject_id')
        if not isinstance(subject_id, str) or not subject_id.strip():
            raise DatasetContractError('Every split record must define a non-empty subject_id.')
        split = record.get('split')
        if split is None or str(split).strip() == '':
            continue
        subject_splits.setdefault(subject_id, set()).add(str(split))
    return {subject: sorted(splits) for subject, splits in subject_splits.items() if len(splits) > 1}


def assert_no_subject_split_leakage(records: Iterable[Mapping]):
    violations = find_subject_split_leakage(records)
    if violations:
        raise DatasetContractError(f'Subject-level data leakage detected: {violations}.')
