import numpy as np
import random

# ฟังก์ชันคำนวณ Mean Absolute Error (MAE)
def calculate_mae(predictions, targets):
    return np.mean(np.abs(predictions - targets))

data = np.random.rand(9358, 15)  # สร้างข้อมูลสุ่มเพื่อเป็นตัวอย่างเท่านั้น

# แบ่งข้อมูลเป็นชุดฝึกและชุดทดสอบ (10% cross-validation)
num_samples = data.shape[0]
num_train_samples = int(0.9 * num_samples)
train_data = data[:num_train_samples, :]
test_data = data[num_train_samples:, :]

# ฟังก์ชันสร้าง MLP
def create_mlp(num_input, num_hidden, num_output):
    # สร้างโครงสร้าง MLP
    network = {
        'input_size': num_input,
        'hidden_size': num_hidden,
        'output_size': num_output,
        'weights_input_hidden': np.random.rand(num_input, num_hidden),
        'weights_hidden_output': np.random.rand(num_hidden, num_output)
    }
    return network

# ฟังก์ชัน Feedforward สำหรับ MLP
def feedforward(network, input_data):
    input_hidden = np.dot(input_data, network['weights_input_hidden'])
    hidden_output = np.dot(input_hidden, network['weights_hidden_output'])
    return hidden_output

def compute_fit(weights, network, input_data, target_data):
    network['weights_input_hidden'] = weights[:network['input_size'] * network['hidden_size']].reshape(
        network['input_size'], network['hidden_size'])
    network['weights_hidden_output'] = weights[network['input_size'] * network['hidden_size']:].reshape(
        network['hidden_size'], network['output_size'])

    predictions = feedforward(network, input_data)

    mae = calculate_mae(predictions, target_data)

    return mae

def particle_swarm_optimization(network, input_data, target_data, num_particles, max_iter):
    num_weights = (network['input_size'] * network['hidden_size'] +
                   network['hidden_size'] * network['output_size'])

    best_weights = np.random.rand(num_weights)
    best_mae = float('inf')
    swarm = [best_weights] * num_particles
    velocities = [np.zeros(num_weights)] * num_particles

    inertia = 0.5
    cognitive_weight = 1.5
    social_weight = 1.5

    for i in range(max_iter):
        for j in range(num_particles):
            mae = compute_fit(swarm[j], network, input_data, target_data)

            if mae < best_mae:
                best_weights = swarm[j]
                best_mae = mae

            new_velocity = (inertia * velocities[j] +
                            cognitive_weight * random.random() * (best_weights - swarm[j]) +
                            social_weight * random.random() * (best_weights - swarm[j]))
            velocities[j] = new_velocity

            swarm[j] += velocities[j]

    return best_mae, best_weights

num_input = 8
num_hidden = 10
num_output = 1

network = create_mlp(num_input, num_hidden, num_output)

num_particles = 20
max_iter = 50
best_mae, best_weights = particle_swarm_optimization(network, train_data[:, [2, 3, 8, 10, 11, 12, 13, 14]], train_data[:, 5], num_particles, max_iter)

test_mae = compute_fit(best_weights, network, test_data[:, [2, 3, 8, 10, 11, 12, 13, 14]], test_data[:, 5])
print(f'Test MAE: {test_mae}')
