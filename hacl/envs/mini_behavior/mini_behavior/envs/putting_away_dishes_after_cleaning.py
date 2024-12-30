from hacl.envs.mini_behavior.mini_behavior.roomgrid import *
from hacl.envs.mini_behavior.mini_behavior.register import register


class PuttingAwayDishesAfterCleaningEnv(RoomGrid):
    """
    Environment in which the agent is instructed to put away dishes after cleaning
    """

    def __init__(
            self,
            mode='primitive',
            room_size=10,
            num_rows=1,
            num_cols=1,
            max_steps=1e5,
            dense_reward=False,
    ):
        num_objs = {'plate': 4, 'countertop': 1, 'cabinet': 1}        

        self.mission = 'put away dishes after cleaning'

        super().__init__(mode=mode,
                         num_objs=num_objs,
                         room_size=room_size,
                         num_rows=num_rows,
                         num_cols=num_cols,
                         max_steps=max_steps,
                         dense_reward=dense_reward,
                         )

    def _gen_objs(self):
        plate = self.objs['plate']
        countertop = self.objs['countertop']
        cabinet = self.objs['cabinet'][0]

        self.place_obj(countertop[0])

        self.place_obj(cabinet)

        countertop_pos = self._rand_subset(countertop[0].all_pos, 5)

        for i in range(4):
            self.put_obj(plate[i], *countertop_pos[i], 1)

    def _end_conditions(self):
        plate = self.objs['plate']
        cabinet = self.objs['cabinet'][0]
        score = 0
        for obj in plate:
            if obj.check_rel_state(self, cabinet, 'inside'):
                score += 0.25
        for obj in plate:
            if not obj.check_rel_state(self, cabinet, 'inside'):
                return False, score

        return True, score

    # This score measures progress towards the goal
    def get_progress(self):
        plate = self.objs['plate']
        cabinet = self.objs['cabinet'][0]
        score = 0
        for obj in plate:
            if obj.check_rel_state(self, cabinet, 'inside'):
                score += 1
        if cabinet.check_abs_state(self, 'openable'):
            score += 1

        return score


class PuttingAwayDishesAfterCleaningGenEnv(RoomGrid):
    """
    Environment in which the agent is instructed to put away dishes after cleaning
    """

    def __init__(
            self,
            mode='primitive',
            room_size=10,
            num_rows=1,
            num_cols=1,
            max_steps=1e5,
            dense_reward=False,
    ):
        num_objs = {'plate': 4, 'countertop': 1, 'cabinet': 1, 'hamburger':1}
        
        self.mission = 'put away dishes after cleaning'

        super().__init__(mode=mode,
                         num_objs=num_objs,
                         room_size=room_size,
                         num_rows=num_rows,
                         num_cols=num_cols,
                         max_steps=max_steps,
                         dense_reward=dense_reward,
                         )

    def _gen_objs(self):
        plate = self.objs['plate']
        countertop = self.objs['countertop']
        cabinet = self.objs['cabinet'][0]
        hamburger = self.objs['hamburger'][0]

        self.place_obj(countertop[0])

        self.place_obj(cabinet)

        countertop_pos = self._rand_subset(countertop[0].all_pos, 5)

        for i in range(4):
            self.put_obj(plate[i], *countertop_pos[i], 1)
        self.put_obj(hamburger, *countertop_pos[4], 1)

    def _end_conditions(self):
        plate = self.objs['plate']
        cabinet = self.objs['cabinet'][0]
        score = 0
        for obj in plate:
            if obj.check_rel_state(self, cabinet, 'inside'):
                score += 0.25
        for obj in plate:
            if not obj.check_rel_state(self, cabinet, 'inside'):
                return False, score

        return True, score

    # This score measures progress towards the goal
    def get_progress(self):
        plate = self.objs['plate']
        cabinet = self.objs['cabinet'][0]
        score = 0
        for obj in plate:
            if obj.check_rel_state(self, cabinet, 'inside'):
                score += 1
        if cabinet.check_abs_state(self, 'openable'):
            score += 1

        return score

# non human input env
register(
    id='MiniGrid-PuttingAwayDishesAfterCleaning-Basic-10x10-N2-v0',
    entry_point='hacl.envs.mini_behavior.mini_behavior.envs:PuttingAwayDishesAfterCleaningEnv'
)

register(
    id='MiniGrid-PuttingAwayDishesAfterCleaning-Gen-10x10-N2-v0',
    entry_point='hacl.envs.mini_behavior.mini_behavior.envs:PuttingAwayDishesAfterCleaningGenEnv'
)