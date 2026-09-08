"""Explicit five-fold runner; defaults to no mode and never trains."""

import argparse

import m4_osseous_formal_protocol as contract


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--contract-audit', action='store_true')
    mode.add_argument('--checkpoint-bind', action='store_true')
    mode.add_argument('--verify-checkpoints', action='store_true')
    mode.add_argument('--evaluate', action='store_true')
    parser.add_argument('--data-root')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'cuda:0'], default='cuda')
    args = parser.parse_args(argv)
    if not args.evaluate and args.data_root:
        parser.error('data-root is forbidden outside explicit evaluation')
    if args.contract_audit:
        result = contract.contract_audit()
    elif args.checkpoint_bind or args.verify_checkpoints:
        from bind_m4_osseous_formal_checkpoints import main as binding_main
        binding_main(['--checkpoint-bind' if args.checkpoint_bind else '--verify-checkpoints'])
        return
    else:
        if not args.data_root:
            parser.error('--evaluate requires --data-root')
        from evaluate_m4_osseous_formal_test import evaluate
        from bind_m4_osseous_formal_checkpoints import read_binding, audit_all
        from aggregate_m4_osseous_formal_test import aggregate
        _, sources = contract.load_protocol()
        binding = read_binding(sources)
        # Preflight every target before allowing the first sample read.
        for fold in sources['clean10']['folds']:
            if contract.safe_path(contract.OUTPUT_ROOT + fold.lower()).exists():
                raise contract.ContractError('a formal fold output already exists')
        if contract.safe_path(contract.OUTPUT_ROOT + 'formal_test_summary.json').exists():
            raise contract.ContractError('formal summary already exists')
        audit_all(sources, binding)
        for fold in sources['clean10']['folds']:
            evaluate(fold, args.data_root, args.device)
        result = aggregate()
    print(contract.canonical(result))


if __name__ == '__main__':
    main()
