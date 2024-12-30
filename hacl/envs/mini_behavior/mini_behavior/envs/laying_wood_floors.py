from hacl.envs.mini_behavior.mini_behavior.roomgrid import *
from hacl.envs.mini_behavior.mini_behavior.register import register


class LayingWoodFloorsEnv(RoomGrid):

    def __init__(
            self,
            mode='primitive',
            room_size=16,
            num_rows=1,
            num_cols=1,
            max_steps=1e5,
    ):
        num_objs = {'plywood': 4, 'hammer': 1, 'saw': 1}
        self.mission = 'laying wood floors'

        super().__init__(mode=mode,
                         num_objs=num_objs,
                         room_size=room_size,
                         num_rows=num_rows,
                         num_cols=num_cols,
                         max_steps=max_steps
                         )

    def _gen_objs(self):
        for obj in self.obj_instances.values():
            self.place_obj(obj)


    def _init_conditions(self):
        for obj_type in ['plywood', 'hammer', 'saw']:
            assert obj_type in self.objs.keys(), f"No {obj_type}"

        return True

    def _end_conditions(self):
        plywood = self.objs['plywood']
        hammer = self.objs['hammer'][0]
        saw = self.objs['saw'][0]

        score = 0
        for wood in plywood:
            nextto = False
            for other_wood in plywood:
                if wood.check_rel_state(self, other_wood, 'nextto'):
                    nextto = True
            if nextto:
                score += 0.25
                
        for wood in plywood:
            if not wood.check_abs_state(self, 'onfloor'):
                return False, score
        
        for wood in plywood:
            nextto = False
            for other_wood in plywood:
                if wood.check_rel_state(self, other_wood, 'nextto'):
                    nextto = True

            if not nextto:
                return False, score

        return True, score
    
    
class LayingWoodFloorsGenEnv(RoomGrid):

    def __init__(
            self,
            mode='primitive',
            room_size=16,
            num_rows=1,
            num_cols=1,
            max_steps=1e5,
    ):
        num_objs = {'plywood': 4, 'hammer': 1, 'saw': 1, 'rag': 1}
        self.mission = 'laying wood floors'

        super().__init__(mode=mode,
                         num_objs=num_objs,
                         room_size=room_size,
                         num_rows=num_rows,
                         num_cols=num_cols,
                         max_steps=max_steps
                         )

    def _gen_objs(self):
        for obj in self.obj_instances.values():
            self.place_obj(obj)


    def _init_conditions(self):
        for obj_type in ['plywood', 'hammer', 'saw']:
            assert obj_type in self.objs.keys(), f"No {obj_type}"

        return True

    def _end_conditions(self):
        plywood = self.objs['plywood']
        hammer = self.objs['hammer'][0]
        saw = self.objs['saw'][0]

        score = 0
        for wood in plywood:
            nextto = False
            for other_wood in plywood:
                if wood.check_rel_state(self, other_wood, 'nextto'):
                    nextto = True
            if nextto:
                score += 0.25
                
        for wood in plywood:
            if not wood.check_abs_state(self, 'onfloor'):
                return False, score
        
        for wood in plywood:
            nextto = False
            for other_wood in plywood:
                if wood.check_rel_state(self, other_wood, 'nextto'):
                    nextto = True

            if not nextto:
                return False, score

        return True, score



# non human input env
register(
    id='MiniGrid-LayingWoodFloors-Basic-16x16-N2-v0',
    entry_point='hacl.envs.mini_behavior.mini_behavior.envs:LayingWoodFloorsEnv'
)

register(
    id='MiniGrid-LayingWoodFloors-Gen-16x16-N2-v0',
    entry_point='hacl.envs.mini_behavior.mini_behavior.envs:LayingWoodFloorsGenEnv'
)

# human input env
# register(
#     id='MiniGrid-LayingWoodFloors-16x16-N2-v1',
#     entry_point='hacl.envs.mini_behavior.mini_behavior.envs:LayingWoodFloorsEnv',
#     kwargs={'mode': 'cartesian'}
# )
