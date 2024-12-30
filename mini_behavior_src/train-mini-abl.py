import os.path as osp
import time
from copy import deepcopy

import numpy as np
import torch
import jacinle
import jacinle.io as io
import jactorch
from jactorch.io import load_state_dict, state_dict

import hacl.pdsketch as pds
import hacl.envs.mini_behavior.minibehavior_v2023 as mb

io.set_fs_verbose()
pds.Value.set_print_options(max_tensor_size=100)
# pds.ConditionalAssignOp.set_options(quantize=True)

parser = jacinle.JacArgumentParser()
parser.add_argument('env', choices=['mini_behavior'])
parser.add_argument('task', choices=mb.SUPPORTED_TASKS)
parser.add_argument('--seed', type=int, default=33)
parser.add_argument('--workers', type=int, default=0)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--update-interval', type=int, default=50)
parser.add_argument('--print-interval', type=int, default=50)
parser.add_argument('--evaluate-interval', type=int, default=250)
parser.add_argument('--save-interval', type=int, default=500)
parser.add_argument('--iterations', type=int, default=1000)
parser.add_argument('--use-offline', type='bool', default=False)
parser.add_argument('--action-loss-weight', type=float, default=1.0)
parser.add_argument('--goal-loss-weight', type=float, default=1.0)

parser.add_argument('--action-mode', type=str, default='envaction', choices=mb.SUPPORTED_ACTION_MODES)
parser.add_argument('--structure-mode', type=str, default='abl', choices=mb.SUPPORTED_STRUCTURE_MODES)
parser.add_argument('--train-heuristic', action='store_true')
parser.add_argument('--load', type='checked_file', default=None)
parser.add_argument('--load-heuristic', type='checked_file', default=None)
parser.add_argument('--discretize', action='store_true')
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--generalize', action='store_true')
parser.add_argument('--visualize', action='store_true')
parser.add_argument('--visualize-failure', action='store_true', help='visualize the tasks that the planner failed to solve.')
parser.add_argument('--evaluate-objects', type=int, default=4)
parser.add_argument('--relevance-analysis', type='bool', default=True)
parser.add_argument('--heuristic', type=str, default='hff', choices=['hff', 'blind', 'external'])

# model-specific options.
parser.add_argument('--model-use-vq', type='bool', default=False, help='only for cw.ModelAbskin2')

# append the results
parser.add_argument('--append-expr', action='store_true')
parser.add_argument('--append-result', action='store_true')

# debug options for neural network training
parser.add_argument('--use-gt-classifier', type='bool', default=False)
parser.add_argument('--use-gt-facing', type='bool', default=False)

# debug options
parser.add_argument('--debug', action='store_true')
parser.add_argument('--debug-domain', action='store_true')
parser.add_argument('--debug-env', action='store_true')
parser.add_argument('--debug-encoder', action='store_true')
parser.add_argument('--debug-action', action='store_true', help='must be used with --debug-encoder')
parser.add_argument('--debug-strips', action='store_true')
parser.add_argument('--debug-ray', action='store_true')
parser.add_argument('--debug-discretize', action='store_true')
args = parser.parse_args()

if args.workers > 0:
    import ray
    ray.init()

if args.env == 'mini_behavior':
    lib = mb
    import hacl.p.kfac.minibehavior.ground_models as lib_models
    if args.task == 'install-a-printer':
        def action_filter(a):
            return a.name in ['forward', 'lturn', 'rturn', 'pickup_0', 'drop_2', 'toggle']
    elif args.task in ['opening_packages', 'opening_packages1', 'opening_packages3']:
        def action_filter(a):
            return a.name in ['forward', 'lturn', 'rturn', 'open']
    elif args.task == 'MovingBoxesToStorage':
        def action_filter(a):
            return a.name in ['forward', 'lturn', 'rturn', 'pickup_0', 'drop_1']
    elif args.task == 'SortingBooks':
        def action_filter(a):
            return a.name in ['forward', 'lturn', 'rturn', 'pickup_0', 'pickup_2', 'drop_2']
    elif args.task in ['Throwing_away_leftovers', 'Throwing_away_leftovers1', 'Throwing_away_leftovers2']:
        def action_filter(a):
            return a.name in ['forward', 'lturn', 'rturn', 'pickup_2', 'drop_in']
    elif args.task == 'PuttingAwayDishesAfterCleaning':
        def action_filter(a):
            return a.name in ['forward', 'lturn', 'rturn', 'pickup_1', 'open', 'drop_in']
    elif args.task == 'BoxingBooksUpForStorage':
        def action_filter(a):
            return a.name in ['forward', 'lturn', 'rturn', 'pickup_0', 'pickup_2', 'drop_in']
    elif args.task == 'CleaningACar':
        def action_filter(a):
            return a.name in ['forward', 'lturn', 'rturn', 'pickup_0', 'pickup_2', 'drop_2', 'drop_in', 'toggle']
    elif args.task == 'CleaningShoes':
        def action_filter(a):
            return a.name in ['forward', 'lturn', 'rturn', 'pickup_0', 'pickup_1', 'drop_1', 'drop_in', 'toggle']
    elif args.task == 'CollectMisplacedItems':
        def action_filter(a):
            return a.name in ['forward', 'lturn', 'rturn', 'pickup_0', 'pickup_1', 'pickup_2', 'drop_2']
    elif args.task == 'LayingWoodFloors':
        def action_filter(a):
            return a.name in ['forward', 'lturn', 'rturn', 'pickup_0', 'drop_0']
    elif args.task == 'MakingTea':
        def action_filter(a):
            return a.name in ['forward', 'lturn', 'rturn', 'pickup_0', 'pickup_1', 'drop_2', 'drop_in', 'open', 'toggle']
    elif args.task == 'OrganizingFileCabinet':
        def action_filter(a):
            return a.name in ['forward', 'lturn', 'rturn', 'pickup_0', 'pickup_1', 'pickup_2', 'drop_2', 'drop_in', 'open']
    elif args.task == 'Washing_pots_and_pans':
        def action_filter(a):
            return a.name in ['forward', 'lturn', 'rturn', 'pickup_0', 'pickup_1', 'drop_in', 'drop_1', 'toggle', 'open']
    elif args.task == 'WateringHouseplants':
        def action_filter(a):
            return a.name in ['forward', 'lturn', 'rturn', 'pickup_0', 'drop_in', 'toggle']
    else:
        raise ValueError(f'Unknown task: {args.task}.')

    from hacl.p.kfac.minibehavior.data_generator import worker_offline
    if args.use_offline:
        worker_inner = worker_offline

else:
    raise ValueError(f'Unknown environment: {args.env}')

Model = lib_models.get_model(args)

assert args.task in lib.SUPPORTED_TASKS
assert args.action_mode in lib.SUPPORTED_ACTION_MODES
assert args.structure_mode in lib.SUPPORTED_STRUCTURE_MODES

logger = jacinle.get_logger(__file__)
log_timestr = time.strftime(time.strftime('%Y-%m-%d-%H-%M-%S'))
if args.load is not None:
    load_id = osp.basename(args.load).split('-')
    if load_id[1] == 'goto' and load_id[2] == 'single':
        load_id = 'gotosingle'
    else:
        load_id = load_id[1]
else:
    load_id = 'scratch'

args.id_string = f'{args.task}-{args.action_mode}-{args.structure_mode}-lr={args.lr}-load={load_id}-{log_timestr}'
args.log_filename = f'dumps/{args.id_string}.log'
args.json_filename = f'dumps/{args.id_string}.jsonl'

if not args.debug:
    jacinle.set_logger_output_file(args.log_filename)

logger.critical('Arguments: {}'.format(args))

if not args.debug and args.append_expr:
    with open('./experiments.txt', 'a') as f:
        f.write(args.id_string + '\n')


def build_model(args, domain, **kwargs):
    return Model(domain, **kwargs)


def worker(index, args, update_model_fn, push_data_fn):
    jacinle.reset_global_seed(index)

    logger.info(f'Worker {index} started. Arguments: {args}')
    logger.critical(f'Worker {index}: creating the domain: action mode = {args.action_mode} structure_mode = {args.structure_mode}')
    lib.set_domain_mode(args.action_mode, args.structure_mode)
    domain = lib.get_domain(force_reload=True)
    logger.critical(f'Worker {index}: creating model...')
    model = build_model(args, domain)
    if args.load is not None:
        load_state_dict(model, io.load(args.load))
    logger.critical(f'Worker {index}: creating environment (task={args.task})...')
    if args.env == 'mini_behavior':
        env = lib.make(args.task)
    else:
        raise ValueError(f'Unknown environment: {args.env}.')

    while True:
        data = worker_inner(args, domain, env, action_filter)
        push_data_fn(data)
        if not update_model_fn(model):
            break


def discretize_feature(domain: pds.Domain, dataset: pds.AODiscretizationDataset, feature_name, input_name=None):
    if input_name is None:
        input_name = feature_name
    mappings = dict()
    for state in dataset.iter_all_states():
        s = state.clone()
        domain.forward_features_and_axioms(s, forward_augmented=True, forward_derived=False, forward_axioms=False)
        input_feature = s[input_name].tensor.reshape(-1, s[input_name].tensor.shape[-1])
        output_feature = s[feature_name].tensor.reshape(-1, s[feature_name].tensor.shape[-1])

        assert input_feature.shape[0] == output_feature.shape[0]
        for i in range(input_feature.shape[0]):
            key = tuple(map(int, input_feature[i]))
            if key not in mappings:
                mappings[key] = output_feature[i]

    return jactorch.as_numpy(torch.stack([mappings[k] for k in sorted(mappings)], dim=0))


def discrete_analysis(domain, env):
    dataset = pds.AODiscretizationDataset()
    from tqdm import tqdm
    for i in tqdm(range(100)):
    # for i in range(1024):
        states, actions, dones, goal, succ, extra_monitors = worker_inner(args, domain, env, action_filter)
        dataset.feed(states, actions, dones, goal)

    a = np.arange(16)
    b = np.arange(1,15)
    if args.env == 'mini_behavior' and args.structure_mode in ('abskin', 'full'):
        manual_feature_discretization = {
            'robot-direction': np.array([[0], [1], [2], [3]], dtype=np.int64),
            'robot-pose': np.stack(np.meshgrid(b, b), axis=-1).reshape(-1, 2),
            'item-pose': np.stack(np.meshgrid(a, a), axis=-1).reshape(-1, 2),
            'item-height': np.array([[0], [1], [2]], dtype=np.int64),
        }
        feature_dims = {'robot-pose': 256, 
                        'item-pose': 256, 
                        'item-type': 32,
                        'item-color': 32,
                        'item-state': 32,
                        'item-height': 16, 
                        'item-feature': 128, 
                        'robot-direction': 16, 
                        'location-pose': 128, 
                        'location-feature': 128, 
                        'robot-feature': 128}
    else:
        manual_feature_discretization = {}
    pds.ao_discretize(domain, dataset, feature_dims, cache_bool_features=True, manual_feature_discretizations=manual_feature_discretization)

    if args.debug_discretize:
        translator = pds.strips.GStripsTranslatorSAS(domain, cache_bool_features=True)
        for i in range(100):
            obs = env.reset()
            env.render()
            task = translator.compile_task(obs['state'], obs['mission'], forward_relevance_analysis=False, backward_relevance_analysis=True, verbose=True)
            relaxed = translator.recompile_relaxed_task(task, forward_relevance_analysis=False, backward_relevance_analysis=True)
            hff = pds.strips.StripsHFFHeuristic(relaxed, translator)
            print(hff.compute(task.state))
            QUIT = False
            from IPython import embed; embed()
            if QUIT:
                break


def evaluate(domain, env, heuristic_model=None, visualize: bool = False):
    def external_heuristic_function(state, goal):
        return heuristic_model.forward(state, goal)

    def planner(state, mission):
        # pds.heuristic_search_strips.DEBUG = True
        args.relevance_analysis = True
        return pds.heuristic_search_strips(
            domain,
            state,
            mission,
            max_depth=20, max_expansions=10000,
            action_filter=action_filter,
            use_tuple_desc=False,
            use_quantized_state=False,
            forward_augmented=True,
            track_most_promising_trajectory=True,
            strips_heuristic=args.heuristic,
            strips_forward_relevance_analysis=False,
            strips_backward_relevance_analysis=args.relevance_analysis,
            strips_use_sas=args.discretize,
            use_strips_op=False,
            verbose=True,
            return_extra_info=not visualize,
            external_heuristic_function=external_heuristic_function,
        )
    if visualize:
        lib.visualize_planner(env, planner)
    else:
        jacinle.reset_global_seed(args.seed, True)
        succ, nr_expansions, total = 0, 0, 0
        nr_expansions_hist = list()
        scores = list()
        for i in jacinle.tqdm(100, desc='Evaluation'):
            score = 0
            obs = env.reset()
            env_copy = deepcopy(env)
            state, goal = obs['state'], obs['mission']
            plan, extra_info = planner(state, goal)
            this_succ = False
            if plan is None:
                pass
            else:
                for action in plan:
                    _, _, (done,score), _ = env.step(action)
                    if done:
                        this_succ = True
                        break
            # done = env.compute_done()
            # if done:
            #     this_succ = True
            scores.append(score)
            nr_expansions += extra_info['nr_expansions']
            succ += int(this_succ)
            total += 1

            print(this_succ, extra_info['nr_expansions'])
            nr_expansions_hist.append({'succ': this_succ, 'expansions': extra_info['nr_expansions']})

            if not this_succ and args.visualize_failure:
                print('Failed to solve the problem.')
                lib.visualize_plan(env_copy, plan=plan)

        # import seaborn as sns
        # sns.histplot(nr_expansions_hist)
        # import matplotlib.pyplot as plt
        # plt.savefig('./visualizations/nr_expansions_hist.png')

        if args.load_heuristic is not None:
            discretize_str = 'learning'
        elif args.heuristic == 'hff':
            discretize_str = 'discretized' if args.discretize else 'no-discretized'
        else:
            assert args.heuristic == 'blind'
            discretize_str = 'blind'
        evaluate_key = f'{args.structure_mode}-{args.task}-load={load_id}-objs={args.evaluate_objects}-{discretize_str}'
        io.dump(f'./results/{evaluate_key}.json', nr_expansions_hist)
        print("avg_score =",sum(scores) / total)
        return succ / total, nr_expansions / total


def main():
    jacinle.reset_global_seed(args.seed, True)
    lib.set_domain_mode(args.action_mode, args.structure_mode)
    domain = lib.get_domain()

    logger.critical('Creating model...')
    model = build_model(
        args,
        domain,
        goal_loss_weight=args.goal_loss_weight,
        action_loss_weight=args.action_loss_weight
    )
    if args.load is not None:
        try:
            load_state_dict(model, io.load(args.load))
        except KeyError as e:
            logger.exception(f'Failed to load (part of the) model: {e}.')

    heuristic_model = None

    # It will be used for evaluation.
    logger.critical(f'Creating environment (task={args.task})...')

    if args.env == 'mini_behavior':
        env = lib.make(args.task, args.generalize)
    else:
        raise ValueError(f'Unknown environment: {args.env}.')
    env.env.seed(args.seed)

    if args.evaluate:
        args.iterations = 0

    data_store = None
    data_workers = list()
    if args.workers > 0:
        logger.warning('Using {} workers.'.format(args.workers))
        import hacl.p.kfac.mprl.workers as W
        data_store = W.DataStore.remote()
        data_workers = [W.DataWorker.remote(i, args, data_store, worker) for i in range(args.workers)]
    else:
        logger.warning('Using single-threaded mode.')

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    meters = jacinle.GroupMeters()
    for i in jacinle.tqdm(range(1, args.iterations + 1)):
        total_n, total_loss = 0, 0
        while total_n < 32:
            end = time.time()
            if args.workers > 0:
                while True:
                    data = ray.get(data_store.pop.remote())
                    if data is not None:
                        break
                    time.sleep(0.1)
                states, actions, dones, goal, succ, extra_monitors = data
            else:
                states, actions, dones, goal, succ, extra_monitors = worker_inner(args, domain, env, action_filter)

            extra_monitors['time/data'] = time.time() - end; end = time.time()
            # print({'states': states, 'actions': actions, 'dones': dones, 'goal_expr': goal})
            if len(actions) <= 0:
                continue
            loss, monitors, output_dict = model(feed_dict={'states': states, 'actions': actions, 'dones': dones, 'goal_expr': goal},task=args.task, forward_augmented=True)
            extra_monitors['time/model'] = time.time() - end; end = time.time()
            extra_monitors['accuracy/succ'] = float(succ)
            monitors.update(extra_monitors)

            n = len(dones)
            total_n += n
            total_loss += loss * n

            meters.update(monitors, n=n)
            jacinle.get_current_tqdm().set_description(
                meters.format_simple(values={k: v for k, v in meters.val.items() if k.count('/') <= 1})
            )

        end = time.time()
        if total_loss.requires_grad:
            opt.zero_grad()
            total_loss /= total_n
            total_loss.backward()
            opt.step()
        meters.update({'time/gd': time.time() - end}, n=1)

        if args.update_interval > 0 and i % args.update_interval == 0:
            if args.workers > 0:
                params = state_dict(model)
                for w in data_workers:
                    w.set_model_parameters.remote(params)

        if args.print_interval > 0 and i % args.print_interval == 0:
            logger.info(meters.format_simple(f'Iteration {i}/{args.iterations}', values='avg', compressed=False))

            if not args.debug:
                with open(args.json_filename, 'a') as f:
                    f.write(jacinle.io.dumps_json({k: float(v) for k, v in meters.avg.items()}) + '\n')

            meters.reset()

        if args.evaluate_interval > 0 and i % args.evaluate_interval == 0:
            succ_rate = evaluate(domain, env, visualize=False)
            logger.critical(f'Iteration {i}/{args.iterations}: succ rate: {succ_rate}')

        if args.save_interval > 0 and i % args.save_interval == 0:
            ckpt_name = f'dumps/{args.structure_mode}-{args.task}-load={load_id}-epoch={i}.pth'
            logger.critical('Saving checkpoint to "{}".'.format(ckpt_name))
            io.dump(ckpt_name, state_dict(model))

    if args.discretize:
        logger.info('Discrete analysis...')
        discrete_analysis(domain, env)

    if args.evaluate:
        if args.env == 'mini_behavior':
            env.set_options(nr_objects=args.evaluate_objects)
            env.env.seed(args.seed+1)
        if not args.visualize:
            succ_rate, avg_expansions = evaluate(domain, env, heuristic_model=heuristic_model, visualize=False)
            if args.append_result:
                with open('./experiments-eval.txt', 'a') as f:
                    print(f'{args.structure_mode}-{args.task}-load={load_id}-objs={args.evaluate_objects},{succ_rate},{avg_expansions}', file=f)
            print(f'succ_rate = {succ_rate}')
            print(f'avg_expansions = {avg_expansions}')
        else:
            evaluate(domain, env, heuristic_model=heuristic_model, visualize=True)
    else:
        ckpt_name = f'dumps/{args.structure_mode}-{args.task}-load={load_id}.pth'
        logger.critical('Saving checkpoint to "{}".'.format(ckpt_name))
        io.dump(ckpt_name, state_dict(model))

    # from IPython import embed; embed()

if __name__ == '__main__':
    main()
