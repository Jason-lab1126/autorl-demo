# Visual robustness: teacher vs. camera-only students

100 episodes per cell, same seed and end-of-episode metric (success = cube ends within 2 cm of the goal). The changes affect only what the camera sees. "Held out" marks conditions outside the randomized student's training ranges (light x0.5 to x2, hues below 0.7, camera offsets up to 10 cm); for the fixed-scene student every change is unseen.

| Change | Held out | Teacher success | Student, fixed scene success | Student, randomized visuals success | Student, fixed scene mean distance [cm] | Student, randomized visuals mean distance [cm] |
|---|---|---|---|---|---|---|
| None | no | 100% | 90% | 1% | 3.34 | 40.88 |
| Light intensity x0.5 | no | 100% | 53% | 0% | 7.72 | 40.98 |
| Light intensity x2 | no | 100% | 0% | 0% | 41.15 | 41.12 |
| Cube color: flat red | no | 100% | 20% | 2% | 28.80 | 40.61 |
| Table color: light wood | no | 100% | 0% | 1% | 41.57 | 40.36 |
| Camera moved 5 cm sideways | no | 100% | 0% | 1% | 43.09 | 40.29 |
| Camera moved 10 cm sideways | no | 100% | 0% | 2% | 40.23 | 40.82 |
| Light intensity x3 | yes | 100% | 0% | 1% | 41.40 | 41.14 |
| Cube color: purple | yes | 100% | 0% | 1% | 40.49 | 40.64 |
| Table color: purple | yes | 100% | 0% | 0% | 41.55 | 41.87 |
| Camera moved 12 cm sideways | yes | 100% | 0% | 0% | 40.56 | 40.86 |
