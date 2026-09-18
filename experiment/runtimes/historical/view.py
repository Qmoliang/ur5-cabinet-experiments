"""View the accepted v4.3/v4.4 recordings; never rerun or alter a controller."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0,str(ROOT/'src/common'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initial',choices=('ellipsoid','sphere'),default='ellipsoid')
    parser.add_argument('--sphere-run',type=Path,default=ROOT/'baselines/v44_sphere')
    parser.add_argument('--ellipsoid-run',type=Path,default=ROOT/'baselines/v43_ellipsoid')
    parser.add_argument('--speed',type=float,default=1.0)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    if args.speed<=0:parser.error('speed must be positive')
    from replay_viewer import FormalComparison
    comparison=FormalComparison(args.sphere_run,args.ellipsoid_run,replay_only=True)
    if args.check:comparison.check()
    else:comparison.run(args.initial,args.speed)


if __name__=='__main__':main()
