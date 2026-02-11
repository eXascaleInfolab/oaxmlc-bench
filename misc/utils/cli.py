import argparse
       
def setup_args(args, cfg):
    '''
    Override config values from parsed CLI arguments.

    :param args: Parsed argparse namespace containing optional overrides.
    :param cfg: Experiment configuration dictionary to update in-place.
    :return: The updated configuration dictionary.
    '''
    if args.device is not None:
        cfg['device'] = args.device
    if args.seed is not None:
        cfg['seed'] = args.seed
    return cfg

def config_main(parser: argparse.ArgumentParser, cfg):
    '''
    Configure CLI flags, parse arguments, and dispatch experiment actions.

    This function mutates ``parser`` by registering all CLI options, updates
    ``cfg`` based on parsed arguments, and runs the appropriate experiment
    path (train, evaluation, completion, or finetune).

    :param parser: Argument parser used to register CLI options.
    :type parser: argparse.ArgumentParser
    :param cfg: Experiment configuration dictionary to update.
    :return: None.
    '''
    parser.add_argument("--seed", type=int, default = None)
    # --evaluate                -> args.evaluate == "both"
    # --evaluate validation     -> args.evaluate == "validation"
    # --evaluate test           -> args.evaluate == "test"
    parser.add_argument("--device", type=str, default = None)
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    me_group = parser.add_mutually_exclusive_group(required=True)
    me_group.add_argument(
        "--evaluate",
        nargs="?",
        const="both",
        default=None,
        choices=["validation", "test", "both"],
        help="Run evaluation. If provided without value, evaluates both validation and test."
    )
    parser.add_argument("--fewshot", action="store_true", default= False, help="Enable fewshot experiment")
    me_group.add_argument("--completion", action="store_true", default= False, help="Enable completion task experiment")
    me_group.add_argument("--finetune", action="store_true", default= False, help="Enable finetuning experiment")
    me_group.add_argument("--train", action = "store_true", default=False, help="Enable training experiment")
    args = parser.parse_args()
    dataset_path = cfg['dataset_path']
    print(args.completion)
    cfg['fewshot_exp'] = args.fewshot or args.finetune
    cfg['completion_exp'] = args.completion
     
    if args.debug:
        print("[debug mode] Enabling postmortem debugging on uncaught exceptions")
        import sys, IPython; sys.excepthook = IPython.core.ultratb.FormattedTB(mode="Context", call_pdb=True)
    from misc.experiment.experiment import Experiment
    # Setup seed and device if provided through command line
    cfg = setup_args(args, cfg)        
    exp = Experiment(cfg)
    print(args.evaluate)
    if args.completion:
        print("> Running completion evaluation")
        exp.main_evaluate_completion(splits=('test',),save_preds={'test'})
    if args.finetune:
        print("> Running finetuning experiment")
        if cfg['method'] in ['dexa', 'ngame']:
            exp.main_train_dexa(only_finetune=True)
        else:
            exp.main_finetune(splits= ('validation', 'test'), save_preds={'test'})
        
    if args.evaluate:
        print(f"> Running evaluation on {args.evaluate} split(s)")
        if args.evaluate == "both":
            exp.main_evaluate(splits=('validation','test'),save_preds={'test'})
        else:
            exp.main_evaluate(splits=(args.evaluate,), save_preds={args.evaluate})
    if args.train:
        print("> Running training")
        if cfg['method'] in ['dexa', 'ngame']:
            exp.main_train_dexa()
        else:
            exp.main_train()
    
        
