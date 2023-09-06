#include <iostream>

#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
#include <cstdlib>

// Função de ativação sigmóide
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Derivada da função sigmóide
double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

class NeuralNetwork {
private:
    int input_size_;
    int hidden_size_;
    int output_size_;
    std::vector<std::vector<double>> weights_ih_;
    std::vector<std::vector<double>> weights_ho_;
    
public:
    NeuralNetwork(int input_size, int hidden_size, int output_size) {
        input_size_ = input_size;
        hidden_size_ = hidden_size;
        output_size_ = output_size;

        // Inicializar pesos aleatórios para as conexões
        srand(static_cast<unsigned>(time(nullptr)));
        weights_ih_.resize(input_size_, std::vector<double>(hidden_size_));
        weights_ho_.resize(hidden_size_, std::vector<double>(output_size_));
        for (int i = 0; i < input_size_; ++i) {
            for (int j = 0; j < hidden_size_; ++j) {
                weights_ih_[i][j] = (rand() % 2000 - 1000) / 1000.0;  // Valores entre -1 e 1
            }
        }
        for (int i = 0; i < hidden_size_; ++i) {
            for (int j = 0; j < output_size_; ++j) {
                weights_ho_[i][j] = (rand() % 2000 - 1000) / 1000.0;
            }
        }
    }

    // Função de treinamento simples
    void train(const std::vector<std::vector<double>>& input, const std::vector<std::vector<double>>& target, int epochs) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            for (size_t i = 0; i < input.size(); ++i) {
                // Forward pass
                std::vector<double> hidden(hidden_size_);
                std::vector<double> output(output_size_);

                for (int j = 0; j < hidden_size_; ++j) {
                    double sum = 0.0;
                    for (int k = 0; k < input_size_; ++k) {
                        sum += input[i][k] * weights_ih_[k][j];
                    }
                    hidden[j] = sigmoid(sum);
                }

                for (int j = 0; j < output_size_; ++j) {
                    double sum = 0.0;
                    for (int k = 0; k < hidden_size_; ++k) {
                        sum += hidden[k] * weights_ho_[k][j];
                    }
                    output[j] = sigmoid(sum);
                }

                // Backpropagation
                std::vector<double> output_error(output_size_);
                for (int j = 0; j < output_size_; ++j) {
                    output_error[j] = target[i][j] - output[j];
                }

                std::vector<double> hidden_error(hidden_size_);
                for (int j = 0; j < hidden_size_; ++j) {
                    double error = 0.0;
                    for (int k = 0; k < output_size_; ++k) {
                        error += output_error[k] * weights_ho_[j][k];
                    }
                    hidden_error[j] = error * sigmoid_derivative(hidden[j]);
                }

                // Atualização dos pesos
                for (int j = 0; j < hidden_size_; ++j) {
                    for (int k = 0; k < output_size_; ++k) {
                        weights_ho_[j][k] += hidden[j] * output_error[k];
                    }
                }

                for (int j = 0; j < input_size_; ++j) {
                    for (int k = 0; k < hidden_size_; ++k) {
                        weights_ih_[j][k] += input[i][j] * hidden_error[k];
                    }
                }
            }
        }
    }

    // Função de inferência
    std::vector<double> predict(const std::vector<double>& input) {
        std::vector<double> hidden(hidden_size_);
        std::vector<double> output(output_size_);

        for (int j = 0; j < hidden_size_; ++j) {
            double sum = 0.0;
            for (int k = 0; k < input_size_; ++k) {
                sum += input[k] * weights_ih_[k][j];
            }
            hidden[j] = sigmoid(sum);
        }

        for (int j = 0; j < output_size_; ++j) {
            double sum = 0.0;
            for (int k = 0; k < hidden_size_; ++k) {
                sum += hidden[k] * weights_ho_[k][j];
            }
            output[j] = sigmoid(sum);
        }

        return output;
    }

};

int main() {
    // Exemplo de uso da rede neural
    int input_size = 2;
    int hidden_size = 4;
    int output_size = 1;

    NeuralNetwork nn(input_size, hidden_size, output_size);

    std::vector<std::vector<double>> input_data = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };
    std::vector<std::vector<double>> target_data = { {0}, {1}, {1}, {0} };

    nn.train(input_data, target_data, 10000);

    for (size_t i = 0; i < input_data.size(); ++i) {
        std::vector<double> input = input_data[i];
        std::vector<double> prediction = nn.predict(input);
        std::cout << "Input: " << input[0] << " " << input[1] << " Prediction: " << prediction[0] << std::endl;
    }

    return 0;
}
