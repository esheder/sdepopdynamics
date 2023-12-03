#!/usr/bin/env python3
import json

if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser(description="Tool to generate json parameter file.")
    p.add_argument('pars', type=float, nargs=5,
                   help="System parameters")
    p.add_argument("mult", type=float, nargs="+",
                   help="Multiplicity vector")
    p.add_argument('-o', type=str, help="Output file")
    args = p.parse_args()
    d = dict((key, v)
             for key, v in zip(("a1", "a2", "b1", "b2", "I"),
                               args.pars)) | {'multiplicity': args.mult}
    if args.o:
        with open(args.o, 'w') as f:
            json.dump(d, f)
    else:
        import pprint
        pprint.pprint(d)
