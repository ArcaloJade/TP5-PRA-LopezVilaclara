import numpy as np
import random
import copy

class particle():

    def __init__(self):
        self.x = (random.random()-0.5)*2  # initial x position
        self.y = (random.random()-0.5)*2 # initial y position
        self.orientation = random.uniform(-np.pi,np.pi) # initial orientation
        self.weight = 1.0

    def set(self, new_x, new_y, new_orientation):
        '''
        set: sets a robot coordinate, including x, y and orientation
        '''
        self.x = float(new_x)
        self.y = float(new_y)
        self.orientation = float(new_orientation)

    def move_odom(self,odom,noise):
        '''
        move_odom: Takes in Odometry data and moves the robot based on the odometry data
        
        Devuelve una particula (del robot) actualizada
        '''      
        delta_rot1 = odom['r1']
        delta_rot2 = odom['r2']
        delta_trans = odom['t']
        alpha1, alpha2, alpha3, alpha4 = noise

        # Aplico el ruido (alphas)
        delta_rot1_hat = delta_rot1 + np.random.normal(0, np.sqrt(alpha1*delta_rot1**2 + alpha2*delta_trans**2))
        delta_trans_hat = delta_trans + np.random.normal(0, np.sqrt(alpha3*delta_trans**2 + alpha4*(delta_rot1**2 + delta_rot2**2)))
        delta_rot2_hat = delta_rot2 + np.random.normal(0, np.sqrt(alpha1*delta_rot2**2 + alpha2*delta_trans**2))

        # Calculo nuevas posiciones y orientación con los deltas
        x_new = self.x + delta_trans_hat * np.cos(self.orientation + delta_rot1_hat)
        y_new = self.y + delta_trans_hat * np.sin(self.orientation + delta_rot1_hat)
        theta_new = self.orientation + delta_rot1_hat + delta_rot2_hat

        # Normalizo theta a [-pi, pi]
        theta_new = (theta_new + np.pi) % (2 * np.pi) - np.pi

        self.set(x_new, y_new, theta_new)

    def set_weight(self, weight):
        '''
        set_weights: sets the weight of the particles
        '''
        #noise parameters
        self.weight  = float(weight)

class RobotFunctions:

    def __init__(self, num_particles=0):
        if num_particles != 0:
            self.num_particles = num_particles
            self.particles = []
            for _ in range(self.num_particles):
                self.particles.append(particle())

            self.weights = np.ones(self.num_particles) / self.num_particles

    def get_weights(self,):
        return self.weights
    
    def get_particle_states(self,):
        samples = np.array([[p.x, p.y, p.orientation] for p in self.particles])
        return samples
    
    def move_particles(self, deltas):
        for part in self.particles:
            part.move_odom(deltas, [0.2, 0.2, 0.001, 0.001])
    
    def get_selected_state(self,):
        '''
        Esta funcion debe devolver lo que ustedes consideran como la posición del robot segun las particulas.
        Queda a su criterio como la obtienen en base a las particulas.
        '''
        states = np.array([[p.x, p.y, p.orientation] for p in self.particles])
        weights = self.get_weights()

        # Media ponderada de x e y
        x_mean = np.average(states[:, 0], weights=weights)
        y_mean = np.average(states[:, 1], weights=weights)

        # Para la orientación, uso media circular
        sin_sum = np.sum(np.sin(states[:, 2]) * weights)
        cos_sum = np.sum(np.cos(states[:, 2]) * weights)
        theta_mean = np.arctan2(sin_sum, cos_sum)

        return [x_mean, y_mean, theta_mean]

    def update_particles(self, data, map_data, grid):
        '''
        La funcion update_particles se llamará cada vez que se recibe data del LIDAR
        Esta funcion toma:
            data: datos del lidar en formato scan (Ver documentacion de ROS sobre tipo de dato LaserScan).
                  Pueden aprovechar la funcion scan_refererence del TP1 para convertir los datos crudos en
                  posiciones globales calculadas
            map_data: Es el mensaje crudo del mapa de likelihood. Pueden consultar la documentacion de ROS
                      sobre tipos de dato OccupancyGrid.
            grid: Es la representación como matriz de numpy del mapa de likelihood. 
                  Importante:
                    - La grilla se indexa como grid[y, x], primero fila (eje Y) y luego columna (eje X).
                    - La celda (0,0) corresponde a la esquina inferior izquierda del mapa en coordenadas de ROS.
        
        Esta funcion debe tomar toda esta data y actualizar el valor de probabilidad (weight) de cada partícula
        En base a eso debe resamplear las partículas. Tenga cuidado al resamplear de hacer un deepcopy para que 
        no sean el mismo objeto de python
        '''
        
        # Info del mapa
        resolution = map_data.info.resolution
        origin_x = map_data.info.origin.position.x
        origin_y = map_data.info.origin.position.y
        width = map_data.info.width
        height = map_data.info.height
        
        # Parámetros del LIDAR
        ranges = data.ranges
        range_min = data.range_min
        range_max = data.range_max
        angle_min = data.angle_min
        angle_max = data.angle_max
        angle_increment = data.angle_increment
        
        # Actualizo los pesos por partícula
        for i, particle in enumerate(self.particles):
            odom_particle = [particle.x, particle.y, particle.orientation]
            
            # Convierto las lecturas LIDAR a coordenadas globales usando la posición de la partícula
            points_map = self.scan_refererence(ranges, range_min, range_max, 
                                            angle_min, angle_max, angle_increment, 
                                            odom_particle)
            
            # Calculo el likelihood de la partícula
            likelihood = 1.0
            valid_points = 0
            
            for j in range(len(points_map[0])):
                x_global = points_map[0][j]
                y_global = points_map[1][j]
                
                # Convierto coordenadas globales a índices del grid
                grid_x = int((x_global - origin_x) / resolution)
                grid_y = int((y_global - origin_y) / resolution)
                
                # Chequeo para que el punto esté dentro del grid
                if 0 <= grid_x < width and 0 <= grid_y < height:
                    # Valor de likelihood del mapa (POR CONSIGNA indexado grid[y, x])
                    likelihood_value = grid[grid_y, grid_x]
                    
                    # Convierto de valor de grilla (0-100) a probabilidad (0-1)
                    prob = likelihood_value / 100.0
                    
                    # Acumulo el likelihood
                    likelihood *= (prob + 0.01)  # El 0.01 lo estoy poniendo para evitar multiplicar por 0
                    valid_points += 1
            
            # Si no hay puntos válidos, asigno peso muy bajo
            if valid_points == 0:
                likelihood = 0.001
            
            # Actualizo el peso de la partícula
            self.particles[i].set_weight(likelihood)
            self.weights[i] = likelihood
        
        # Normalizo los pesos
        weight_sum = np.sum(self.weights)
        if weight_sum > 0:
            self.weights = self.weights / weight_sum
        else:
            # Si todos los pesos son 0, usar distribución uniforme (pasó algo raro!!!)
            self.weights = np.ones(self.num_particles) / self.num_particles
        
        # Resampleo basado en pesos
        new_particles = []
        
        # Índices de resampleo
        indices = np.random.choice(self.num_particles, self.num_particles, p=self.weights)
        
        # Creo nuevas partículas HACIENDO DEEPCOPY (importante!)
        for idx in indices:
            new_particle = copy.deepcopy(self.particles[idx])
            new_particles.append(new_particle)
        
        # Actualizo las partículas
        self.particles = new_particles
        
        # Reinicio pesos a distribución uniforme después del resampleo
        self.weights = np.ones(self.num_particles) / self.num_particles


    def scan_refererence(self, ranges, range_min, range_max, angle_min, angle_max, angle_increment, last_odom):
        '''
        Scan Reference recibe:
            - ranges: lista rangos del escáner láser
            - range_min: rango mínimo del escáner
            - range_max: rango máximo del escáner
            - angle_min: ángulo mínimo del escáner
            - angle_max: ángulo máximo del escáner
            - angle_increment: incremento de ángulo del escáner
            - last_odom: última odometría [tx, ty, theta]
        Devuelve puntos en el mapa transformados a coordenadas globales donde 
            - points_map[0]: coordenadas x
            - points_map[1]: coordenadas y
        '''
        
        x_odom, y_odom, theta_odom = last_odom
    
        points_x = []
        points_y = []

        for i, r in enumerate(ranges):
            if range_min <= r <= range_max:
                angle = angle_min + i * angle_increment + np.pi # El pi son los 180 para corregir la orientación (lo dijo Tadeo)
                # Coordenadas locales
                x_local = r * np.cos(angle)
                y_local = r * np.sin(angle)
                # Transformación a global
                x_global = x_odom + x_local * np.cos(theta_odom) - y_local * np.sin(theta_odom)
                y_global = y_odom + x_local * np.sin(theta_odom) + y_local * np.cos(theta_odom)
                points_x.append(x_global)
                points_y.append(y_global)
        
        points_map = np.array([np.array(points_x), np.array(points_y)])
        return points_map
