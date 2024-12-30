from hacl.envs.mini_behavior.mini_behavior.roomgrid import *
from hacl.envs.mini_behavior.mini_behavior.register import register


class BoxingBooksUpForStorageEnv(RoomGrid):
    """
    Environment in which the agent is instructed to clean a car
    """

    def __init__(
            self,
            mode='primitive',
            room_size=10,
            num_rows=1,
            num_cols=1,
            max_steps=1e5,
    ):
        num_objs = {'book': 5, 'shelf': 1, 'box': 1}

        self.mission = 'box up books for storage'

        super().__init__(mode=mode,
                         num_objs=num_objs,
                         room_size=room_size,
                         num_rows=num_rows,
                         num_cols=num_cols,
                         max_steps=max_steps
                         )

        box = self.objs['box'][0]
        box.width = 3
        box.height = 3

    def _gen_objs(self):
        book = self.objs['book']
        shelf = self.objs['shelf'][0]
        box = self.objs['box'][0]
        self.place_obj(shelf)
        self.place_obj(box)

        for obj in book[:3]:
            self.place_obj(obj)

        shelf_pos = self._rand_subset(shelf.all_pos, 2)
        self.put_obj(book[3], *shelf_pos[0], 2)
        self.put_obj(book[4], *shelf_pos[1], 2)


    def _end_conditions(self):
        book = self.objs['book']
        box = self.objs['box'][0]
        score = 0
        for obj in book:
            if obj.check_rel_state(self, box, 'inside'):
                score += 0.2
                
        for obj in book:
            if not obj.check_rel_state(self, box, 'inside'):
                return False, score

        return True, score


class BoxingBooksUpForStorageGenEnv(RoomGrid):
    """
    Environment in which the agent is instructed to clean a car
    """

    def __init__(
            self,
            mode='primitive',
            room_size=10,
            num_rows=1,
            num_cols=1,
            max_steps=1e5,
    ):
        num_objs = {'book': 5, 'shelf': 1, 'box': 1, 'hardback': 1}

        self.mission = 'box up books for storage'

        super().__init__(mode=mode,
                         num_objs=num_objs,
                         room_size=room_size,
                         num_rows=num_rows,
                         num_cols=num_cols,
                         max_steps=max_steps
                         )

        box = self.objs['box'][0]
        box.width = 3
        box.height = 3

    def _gen_objs(self):
        book = self.objs['book']
        shelf = self.objs['shelf'][0]
        box = self.objs['box'][0]
        hardback = self.objs['hardback'][0]
        self.place_obj(shelf)
        self.place_obj(box)

        for obj in book[:3]:
            self.place_obj(obj)

        shelf_pos = self._rand_subset(shelf.all_pos, 3)
        self.put_obj(book[3], *shelf_pos[0], 2)
        self.put_obj(book[4], *shelf_pos[1], 2)
        self.put_obj(hardback, *shelf_pos[2], 2)


    def _end_conditions(self):
        book = self.objs['book']
        box = self.objs['box'][0]
        score = 0
        for obj in book:
            if obj.check_rel_state(self, box, 'inside'):
                score += 0.2
                
        for obj in book:
            if not obj.check_rel_state(self, box, 'inside'):
                return False, score

        return True, score


# non human input env
register(
    id='MiniGrid-BoxingBooksUpForStorage-Basic-10x10-N2-v0',
    entry_point='hacl.envs.mini_behavior.mini_behavior.envs:BoxingBooksUpForStorageEnv'
)

register(
    id='MiniGrid-BoxingBooksUpForStorage-Gen-10x10-N2-v0',
    entry_point='hacl.envs.mini_behavior.mini_behavior.envs:BoxingBooksUpForStorageGenEnv'
)