"""Ravens tasks."""

from hacl.envs.cliport.cliport.tasks.assembling_kits import AssemblingKits
from hacl.envs.cliport.cliport.tasks.packing_5shapes import Packing5Shapes
from hacl.envs.cliport.cliport.tasks.packing_shapes import PackingShapes

from hacl.envs.cliport.cliport.tasks.place_red_in_green import PlaceRedInGreen
from hacl.envs.cliport.cliport.tasks.put_block_in_bowl import PutBlockInBowlSeenColors
from hacl.envs.cliport.cliport.tasks.put_block_in_bowl import PutBlockInBowlComposedColors

from hacl.envs.cliport.cliport.tasks.separating_piles import SeparatingPilesSeenColors
from hacl.envs.cliport.cliport.tasks.separating_piles import Separating20Piles
from hacl.envs.cliport.cliport.tasks.separating_piles import Separating10Piles

names = {
    # goal conditioned
    'assembling-kits': AssemblingKits,
    'place-red-in-green': PlaceRedInGreen,
    'packing-5shapes': Packing5Shapes,
    'packing-shapes': PackingShapes,
    'put-block-in-bowl-seen-colors': PutBlockInBowlSeenColors,
    'put-block-in-bowl-composed-colors': PutBlockInBowlComposedColors,
    'separating-piles-seen-colors': SeparatingPilesSeenColors,
    'separating-20piles': Separating20Piles,
    'separating-10piles': Separating10Piles,
}
