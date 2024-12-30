from hacl.envs.mini_behavior.mini_behavior.roomgrid import *
from hacl.envs.mini_behavior.mini_behavior.register import register


class OpeningPackagesEnv(RoomGrid):
    """
    Environment in which the agent is instructed to clean a car
    """

    def __init__(
            self,
            mode='primitive',
            room_size=16,
            num_rows=1,
            num_cols=1,
            max_steps=1e5,
            num_objs=None
    ):
        if num_objs is None:
            num_objs = {'package': 2}
            #num_objs = {'package': 2, 'basket':1}
            

        self.mission = 'open packages'

        super().__init__(mode=mode,
                         num_objs=num_objs,
                         room_size=room_size,
                         num_rows=num_rows,
                         num_cols=num_cols,
                         max_steps=max_steps
                         )

    def _gen_objs(self):
        for package in self.objs['package']:
            self.place_obj(package)
            package.states['openable'].set_value(False)
        #basket = self.objs['basket'][0]
        #self.place_obj(basket)

    def _init_conditions(self):
        for obj_type in ['package']:
            assert obj_type in self.objs.keys(), f"No {obj_type}"

        for package in self.objs['package']:
            assert not package.check_abs_state(self, 'openable')

        return True

    def _end_conditions(self):
        score = 0
        for package in self.objs['package']:
            if package.check_abs_state(self, 'openable'):
                score += 1/len(self.objs['package'])
        for package in self.objs['package']:
            if not package.check_abs_state(self, 'openable'):
                return False, score
        return True, score
    
class OpeningPackagesGenEnv(RoomGrid):
    """
    Environment in which the agent is instructed to clean a car
    """

    def __init__(
            self,
            mode='primitive',
            room_size=16,
            num_rows=1,
            num_cols=1,
            max_steps=1e5,
            num_objs=None
    ):
        if num_objs is None:
            num_objs = {'package': 2}
            num_objs = {'package': 2, 'basket':1}
            
        self.mission = 'open packages'

        super().__init__(mode=mode,
                         num_objs=num_objs,
                         room_size=room_size,
                         num_rows=num_rows,
                         num_cols=num_cols,
                         max_steps=max_steps
                         )

    def _gen_objs(self):
        for package in self.objs['package']:
            self.place_obj(package)
            package.states['openable'].set_value(False)
        basket = self.objs['basket'][0]
        self.place_obj(basket)

    def _init_conditions(self):
        for obj_type in ['package']:
            assert obj_type in self.objs.keys(), f"No {obj_type}"

        for package in self.objs['package']:
            assert not package.check_abs_state(self, 'openable')

        return True

    def _end_conditions(self):
        score = 0
        for package in self.objs['package']:
            if package.check_abs_state(self, 'openable'):
                score += 1/len(self.objs['package'])
        for package in self.objs['package']:
            if not package.check_abs_state(self, 'openable'):
                return False, score
        return True, score


# non human input env
register(
    id='MiniGrid-OpeningPackages-Basic-16x16-N2-v0',
    entry_point='hacl.envs.mini_behavior.mini_behavior.envs:OpeningPackagesEnv'
)

register(
    id='MiniGrid-OpeningPackages-Gen-16x16-N2-v0',
    entry_point='hacl.envs.mini_behavior.mini_behavior.envs:OpeningPackagesGenEnv'
)

register(
    id='MiniGrid-OpeningPackages1-16x16-N2-v0',
    entry_point='hacl.envs.mini_behavior.mini_behavior.envs:OpeningPackagesEnv',
    kwargs={'num_objs': {'package': 1}}
)

register(
    id='MiniGrid-OpeningPackages3-16x16-N2-v0',
    entry_point='hacl.envs.mini_behavior.mini_behavior.envs:OpeningPackagesEnv',
    kwargs={'num_objs': {'package': 3}}
)
