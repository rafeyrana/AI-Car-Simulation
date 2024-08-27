
import math
import sys
import neat
import pygame
import pickle

WIDTH = 1920
HEIGHT = 1080

CAR_SIZE_X = 60    
CAR_SIZE_Y = 60

BORDER_COLOR = (255, 255, 255, 255) # Color To Crash on Hit

current_generation = 0 # Generation counter

class Car:

    def __init__(self):
        self._load_sprite()
        self._set_default_position()

        self.angle = 0
        self.speed = 0
        self.speed_set = False

        self.center = self._calculate_center()

        self.radars = []
        self.drawing_radars = []

        self.alive = True
        self.distance = 0
        self.time = 0

    def _load_sprite(self):
        self.sprite = pygame.image.load('car.png').convert()
        self.sprite = pygame.transform.scale(self.sprite, (CAR_SIZE_X, CAR_SIZE_Y))
        self.rotated_sprite = self.sprite

    def _set_default_position(self):
        self.position = [830, 920]

    def _calculate_center(self):
        return [
            self.position[0] + CAR_SIZE_X / 2,
            self.position[1] + CAR_SIZE_Y / 2
        ]

    def draw(self, screen):
        screen.blit(self.rotated_sprite, self.position) # Draw Sprite
        self.draw_radar(screen) #OPTIONAL FOR SENSORS

    def draw_radar(self, screen):
        # Optionally Draw All Sensors / Radars
        for radar in self.radars:
            position = radar[0]
            pygame.draw.line(screen, (240, 255, 0), self.center, position, 1)
            pygame.draw.circle(screen, (240, 255, 0), position, 5)

    def check_collision(self, game_map):
        self.alive = True
        for point in self.corners:
            # If Any Corner Touches Border Color -> Crash
            # Assumes Rectangle
            if game_map.get_at((int(point[0]), int(point[1]))) == BORDER_COLOR:
                self.alive = False
                break

    def check_radar(self, degree: int, game_map: pygame.Surface) -> None:
      
        start_x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * 0)
        start_y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * 0)
        max_length = 300
        length = 0
        while length < max_length and not game_map.get_at((start_x, start_y)) == BORDER_COLOR:
            length += 1
            start_x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
            start_y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

        dist = int(math.sqrt(math.pow(start_x - self.center[0], 2) + math.pow(start_y - self.center[1], 2)))
        self.radars.append([(start_x, start_y), dist])
        
    def update(self, game_map):
        self._update_position()
        self._check_collision(game_map)
        self._check_radars(game_map)

    def _update_position(self):
        if not self.speed_set:
            self.speed = 20
            self.speed_set = True

        self.rotated_sprite = self.rotate_center(self.sprite, self.angle)

        x = self.position[0] + math.cos(math.radians(360 - self.angle)) * self.speed
        x = max(x, 20)
        x = min(x, WIDTH - 120)
        self.position[0] = x

        y = self.position[1] + math.sin(math.radians(360 - self.angle)) * self.speed
        y = max(y, 20)
        y = min(y, WIDTH - 120)
        self.position[1] = y

        self.distance += self.speed
        self.time += 1

        self.center = [
            int(self.position[0]) + CAR_SIZE_X / 2,
            int(self.position[1]) + CAR_SIZE_Y / 2
        ]

    def _check_collision(self, game_map):
        self.alive = True
        for point in self._get_corners():
            if game_map.get_at((int(point[0]), int(point[1]))) == BORDER_COLOR:
                self.alive = False
                break

    def _get_corners(self):
        length = 0.5 * CAR_SIZE_X
        left_top = [
            self.center[0] + math.cos(math.radians(360 - (self.angle + 30))) * length,
            self.center[1] + math.sin(math.radians(360 - (self.angle + 30))) * length
        ]
        right_top = [
            self.center[0] + math.cos(math.radians(360 - (self.angle + 150))) * length,
            self.center[1] + math.sin(math.radians(360 - (self.angle + 150))) * length
        ]
        left_bottom = [
            self.center[0] + math.cos(math.radians(360 - (self.angle + 210))) * length,
            self.center[1] + math.sin(math.radians(360 - (self.angle + 210))) * length
        ]
        right_bottom = [
            self.center[0] + math.cos(math.radians(360 - (self.angle + 330))) * length,
            self.center[1] + math.sin(math.radians(360 - (self.angle + 330))) * length
        ]
        return [left_top, right_top, left_bottom, right_bottom]

    def _check_radars(self, game_map):
        self.radars.clear()
        for degree in range(-90, 120, 45):
            self.check_radar(degree, game_map)

    def get_data(self):
        radars = self.radars
        return_values = [0, 0, 0, 0, 0]
        for i, radar in enumerate(radars):
            return_values[i] = int(radar[1] / 30)

        return return_values
    def is_alive(self):
        return self.alive

    def get_reward(self):
        return self.distance / (CAR_SIZE_X / 2)

    def rotate_center(self, image, angle):
        rectangle = image.get_rect()
        rotated_image = pygame.transform.rotate(image, angle)
        rotated_rectangle = rectangle.copy()
        rotated_rectangle.center = rotated_image.get_rect().center
        rotated_image = rotated_image.subsurface(rotated_rectangle).copy()
        return rotated_image



def run_simulation(genomes, config):
    
    nets = []
    cars = []

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)

    for i, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0

        cars.append(Car())

    clock = pygame.time.Clock()
    generation_font = pygame.font.SysFont("Arial", 30)
    alive_font = pygame.font.SysFont("Arial", 20)
    game_map = pygame.image.load('map3.png').convert() 

    global current_generation
    current_generation += 1

    counter = 0

    while True:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)

        for i, car in enumerate(cars):
            output = nets[i].activate(car.get_data())
            choice = output.index(max(output))
            if choice == 0:
                car.angle += 10 # Left
            elif choice == 1:
                car.angle -= 10 # Right
            elif choice == 2:
                if(car.speed - 2 >= 12):
                    car.speed -= 2 # Slow Down
            else:
                car.speed += 2 # Speed Up
        
        # Check If Car Is Still Alive
        # Increase Fitness If Yes And Break Loop If Not
        still_alive = 0
        for i, car in enumerate(cars):
            if car.is_alive():
                still_alive += 1
                car.update(game_map)
                genomes[i][1].fitness += car.get_reward()

        if still_alive == 0:
            break

        counter += 1
        if counter == 30 * 40: # Stop After About 20 Seconds
            break

        # Draw Map And All Cars That Are Alive
        screen.blit(game_map, (0, 0))
        for car in cars:
            if car.is_alive():
                car.draw(screen)
        
        # Display Info
        text = generation_font.render("Generation: " + str(current_generation), True, (0,0,0))
        text_rect = text.get_rect()
        text_rect.center = (900, 450)
        screen.blit(text, text_rect)

        text = alive_font.render("Still Alive: " + str(still_alive), True, (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (900, 490)
        screen.blit(text, text_rect)

        pygame.display.flip()
        clock.tick(60) # 60 FPS




def run_simulation_with_loaded_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    car = Car()
    
    # Set up PyGame and other necessary components
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
    clock = pygame.time.Clock()
    game_map = pygame.image.load('map2.png').convert()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return

        output = net.activate(car.get_data())
        choice = output.index(max(output))
        # Update car based on choice
        
        car.update(game_map)
        
        # Draw everything
        screen.blit(game_map, (0, 0))
        car.draw(screen)
        
        pygame.display.flip()
        clock.tick(60)




if __name__ == "__main__":
    config_path = "./config.txt"
    config = neat.config.Config(neat.DefaultGenome,
                                neat.DefaultReproduction,
                                neat.DefaultSpeciesSet,
                                neat.DefaultStagnation,
                                config_path)

    # with open('best_genome.pkl', 'rb') as f:
    #     loaded_genome = pickle.load(f)

    #     run_simulation_with_loaded_genome(loaded_genome, config)


    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    
    population.run(run_simulation, 1000)


    # best_genome = max(population.population.values(), key=lambda g: g.fitness)
    # with open('best_genome.pkl', 'wb') as f:
    #     pickle.dump(best_genome, f)

  