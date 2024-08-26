# F1-car-simulation
### As a Formula 1 fan, I'm in awe of the drivers' ability to achieve such precision and tight margins on the track. I can't drive an F1 car (yet), but I can code one! So, I'm hand-drawing different maps inspired by the F1 calendar and letting my AI F1 racer take the wheel. 
### Implemented using NEAT (NeuroEvolution of Augmenting Topologies), NEAT is an algorithm for evolving artificial neural networks using: 
- **Genetic Encoding**: Each neural network is encoded as a genome, which includes nodes and connections.
- **Crossover and Mutation**: Genomes can undergo crossover (combining parts of two genomes) and mutation (random changes) to produce offspring.
- **Speciation**: Genomes are grouped into species based on their structural similarity, which helps maintain diversity in the population.
- **Innovation Numbers**: Each new structure (node or connection) is assigned a unique innovation number, allowing for effective crossover between different topologies.

The neural network uses a radar system input which then passes through a hidden layer to 4 output neurons which correspond to Left, Right, Speed Up, Slow Down.
For each action we implement a reward system based on the fitness metric. After each generation we keep the top 2 fittest cars from the last generation and evolve them until they reach max number of generations or complete 10 laps without crashing. 

