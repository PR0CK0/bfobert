import pandas as pd
import random

continuants = [
  'tree','mountain','car','electron','organism','city','heart','rock','book','river',
  'planet','atom','person','knife','bridge','machine','protein','sand','bottle','house',
  'ocean','cell','wheel','desk','forest','chair','tool','bone','metal','island',
  'painting','shoe','tissue','engine','leaf','computer','building','cloud','furniture','mirror',
  'muscle','table','road','pipe','plant','skull','fish','bird','flower','valve',
  'brain','statue','glove','helmet','ship','mountain range','guitar','soil','coat','coin'
]

occurrents = [
  'growth','movement','melting','explosion','conversation','storm','thinking','birth','decay','healing',
  'migration','running','erosion','singing','eating','war','dance','falling','training','flight',
  'reproduction','meeting','rotation','communication','collision','breathing','sleep','decision','transport','oxidation',
  'charging','infection','digestion','evaporation','measurement','reaction','construction','writing','reading','exercise',
  'walking','sleeping','talking','computation','translation','collapse','illumination','competition','speaking','cooking'
]

adjectives = [
  'blue','ancient','rapid','neural','mechanical','chemical','digital','social','thermal','atomic',
  'biological','cosmic','geological','musical','political','genetic','aerial','structural','organic','plastic',
  'marine','industrial','solar','hydraulic','lunar','environmental','robotic','magnetic','artificial','subterranean',
  'cellular','electronic','radiant','tropical','urban','rural','cultural','vocal','linguistic','symbolic'
]

def generateExpanded(baseTerms, targetCount):
  generated = set()
  max_attempts = targetCount * 100  # Allow 100 attempts per desired term
  attempts = 0

  while len(generated) < targetCount and attempts < max_attempts:
      term = random.choice(baseTerms)
      combo = term
      if random.random() < 0.7:
          combo = random.choice(adjectives) + ' ' + combo
      if random.random() < 0.3:
          combo = random.choice(adjectives) + ' ' + combo
      generated.add(combo)
      attempts += 1

  if len(generated) < targetCount:
      print(f"Warning: Only generated {len(generated)} unique terms (requested {targetCount})")

  return list(generated)

continuants = generateExpanded(continuants, 1000)
occurrents = generateExpanded(occurrents, 1000)

data = pd.DataFrame({
  'term': continuants + occurrents,
  'label': [0] * len(continuants) + [1] * len(occurrents)
}).sample(frac = 1).reset_index(drop = True)

data.to_csv('terms.csv', index = False)
print('Generated dataset:', data.shape)
print(data.head())
