from hacl.envs.mini_behavior.mini_behavior.roomgrid import *
from hacl.envs.mini_behavior.mini_behavior.register import register


class MovingBoxesToStorageEnv(RoomGrid):
    """
    Environment in which the agent is instructed to clean a car
    """

    def __init__(
            self,
            mode='primitive',
            room_size=16,
            max_steps=1e5,
    ):
        num_objs = {'carton': 2, 'shelf': 1}        

        self.mission = 'move boxes to storage'

        super().__init__(mode=mode,
                         num_objs=num_objs,
                         room_size=room_size,
                         num_rows=1,
                         num_cols=2,
                         max_steps=max_steps
                         )

    def _gen_objs(self):
        cartons = self.objs['carton']
        shelf = self.objs['shelf'][0]

        self.place_in_room(0, 0, cartons[0])
        self.place_in_room(1, 0, cartons[1])
        self.place_in_room(1, 0, shelf)

        # agent start in room 2

    def _init_conditions(self):
        for obj_type in ['carton', 'shelf']:
            assert obj_type in self.objs.keys(), f"No {obj_type}"

        return True

    def _end_conditions(self):
        cartons = self.objs['carton']

        if cartons[0].check_abs_state(self, 'onfloor') and cartons[1].check_rel_state(self,cartons[0], 'onTop'):
            return True, 1

        if cartons[1].check_abs_state(self, 'onfloor') and cartons[0].check_rel_state(self,cartons[1], 'onTop'):
            return True, 1

        return False, 0
    
class MovingBoxesToStorageGenEnv(RoomGrid):
    """
    Environment in which the agent is instructed to clean a car
    """

    def __init__(
            self,
            mode='primitive',
            room_size=16,
            max_steps=1e5,
    ):
        num_objs = {'carton': 2, 'shelf': 1, 'basket':1}
        
        self.mission = 'move boxes to storage'

        super().__init__(mode=mode,
                         num_objs=num_objs,
                         room_size=room_size,
                         num_rows=1,
                         num_cols=2,
                         max_steps=max_steps
                         )

    def _gen_objs(self):
        cartons = self.objs['carton']
        shelf = self.objs['shelf'][0]
        basket = self.objs['basket']

        self.place_in_room(0, 0, cartons[0])
        self.place_in_room(1, 0, cartons[1])
        self.place_in_room(0, 0, basket[0])
        self.place_in_room(1, 0, shelf)

        # agent start in room 2

    def _init_conditions(self):
        for obj_type in ['carton', 'shelf']:
            assert obj_type in self.objs.keys(), f"No {obj_type}"

        return True

    def _end_conditions(self):
        cartons = self.objs['carton']

        if cartons[0].check_abs_state(self, 'onfloor') and cartons[1].check_rel_state(self,cartons[0], 'onTop'):
            return True, 1

        if cartons[1].check_abs_state(self, 'onfloor') and cartons[0].check_rel_state(self,cartons[1], 'onTop'):
            return True, 1

        return False, 0


# non human input env
register(
    id='MiniGrid-MovingBoxesToStorage-Basic-16x16-N2-v0',
    entry_point='hacl.envs.mini_behavior.mini_behavior.envs:MovingBoxesToStorageEnv',
    kwargs={"room_size": 16}
)

register(
    id='MiniGrid-MovingBoxesToStorage-Gen-16x16-N2-v0',
    entry_point='hacl.envs.mini_behavior.mini_behavior.envs:MovingBoxesToStorageGenEnv',
    kwargs={"room_size": 16}
)

