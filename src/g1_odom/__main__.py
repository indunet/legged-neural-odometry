"""Command line entry point: ``python -m g1_odom <command>`` or ``g1-odom <command>``."""
import argparse
import runpy
import sys

from .common import ROOT

COMMANDS = {'fetch': None, 'collect': 'dataset', 'run': 'workflow', 'serve': 'serve'}


def main():
    parser = argparse.ArgumentParser(prog='g1-odom', description='G1 planar neural odometry')
    parser.add_argument('command', choices=list(COMMANDS))
    args, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0], *rest]
    if args.command == 'fetch':
        runpy.run_path(str(ROOT/'scripts/fetch_assets.py'), run_name='__main__')
    else:
        runpy.run_module('g1_odom.' + COMMANDS[args.command], run_name='__main__')


if __name__ == '__main__':
    main()
