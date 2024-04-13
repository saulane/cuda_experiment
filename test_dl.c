#include <stdio.h>
#include <math.h>

// Structure to hold the parameters of a dense layer
typedef struct
{
    double *weights; // Array to store weights
    double *biases;  // Array to store biases
    int input_size;
    int output_size;
} DenseLayer;

// Sigmoid activation function
void forwardSigmoid(double *x)
{
    x[0] = 1 / (1 + exp(-x[0]));
}

void backwardSigmoid(double *x)
{
    x[0] = exp(-x[0]) / (powf(exp(-x[0] + 1), 2.0f));
}

double MSELoss(double *predictions, double *targets, int n)
{
    double loss = 0;
    for (int i = 0; i < n; i++)
    {
        loss += powf(predictions[i] - targets[i], 2.0f);
    }
    return loss / n;
}

void gradMSE(DenseLayer *layer, double *targets, double *inputs, double *outputs, double *W, double *B, double *d_inputs, double *d_w, double *d_b)
{
    double d_output = 0;
    for (int i = 0; i < layer->output_size; i++)
    {
        d_output = 2 * (outputs[i] - targets[i]) / layer->output_size;

        for (int j = 0; j < layer->input_size; j++)
        {
            // Gradient w.r.t. weights
            d_w[i * layer->input_size + j] = d_output * inputs[j];
            // Gradient w.r.t. input
            d_inputs[j] += W[i * layer->input_size + j] * d_output;
        }

        // Gradient w.r.t. bias
        d_b[i] = d_output;
    }
}

// Performs the forward pass of the dense layer
void forwardLinear(DenseLayer *layer, double *inputs, double *outputs)
{
    for (int i = 0; i < layer->output_size; i++)
    {
        outputs[i] = 0;
        // Perform the dot product of weights and inputs
        for (int j = 0; j < layer->input_size; j++)
        {
            outputs[i] += layer->weights[i * layer->input_size + j] * inputs[j];
        }
        // Add bias and apply sigmoid activation
        outputs[i] = outputs[i] + layer->biases[i];
    }
}

// Backward pass
// void backward(DenseLayer *layer, double *d_outputs, double *d_inputs)
// {
//     for (int i = 0; i < layer->input_size; i++)
//     {
//         d_inputs[i] = 0;
//     }

//     for (int i = 0; i < layer->output_size; i++)
//     {
//         double d_output = d_outputs[i] * sigmoid_derivative(layer->outputs[i]);

//         for (int j = 0; j < layer->input_size; j++)
//         {
//             // Gradient w.r.t. weights
//             double d_weight = d_output * layer->inputs[j];
//             // Update weights with gradient descent
//             layer->weights[i * layer->input_size + j] -= learning_rate * d_weight;

//             // Gradient w.r.t. input
//             d_inputs[j] += layer->weights[i * layer->input_size + j] * d_output;
//         }

//         // Gradient w.r.t. bias
//         double d_bias = d_output;
//         // Update bias with gradient descent
//         layer->biases[i] -= learning_rate * d_bias;
//     }
// }

void zero_grad(double *grad)
{
    for (int i = 0; i < sizeof(grad) / sizeof(double); i++)
    {
        grad[i] = 0.0f;
    }
}

// Example usage
int main()
{
    // Initialize the dense layer
    int input_size = 3;                 // Example input size
    int output_size = 1;                // Example output size
    double weights[] = {0.1, 0.2, 0.3}; // Example weights
    double biases[] = {
        0.1,
    }; // Example biases

    double inputs[] = {1.0f, 2.0f, 3.0f}; // Example input vector
    double targets[1] = {2.0f};
    double outputs[1]; // Output array to store the forward pass results

    DenseLayer layer = {
        weights,
        biases,
        input_size,
        output_size};

    double d_inputs[3] = {0.0f};
    double d_w[3] = {0.0f};
    double d_b[1] = {0.0f};

    int EPOCHS = 10;
    double LR = 0.001;

    for (int i = 0; i < EPOCHS; i++)
    {

        printf("\n-----------------EPOCH %d-----------------\n", i);
        zero_grad(d_inputs);
        zero_grad(d_w);
        zero_grad(d_b);
        forwardLinear(&layer, inputs, outputs);
        double loss = MSELoss(outputs, targets, output_size);

        printf("Loss epoch %d: %f\n", i, loss);
        gradMSE(&layer, targets, inputs, outputs, weights, biases, d_inputs, d_w, d_b);

        // Gradient Descent
        printf("Weights / bias: ");
        for (int j = 0; j < input_size; j++)
        {
            weights[j] -= LR * d_w[j];
            printf(" %f", weights[j]);
        }

        printf("/");
        for (int j = 0; j < output_size; j++)
        {
            biases[j] -= LR * d_b[j];
            printf(" %f", biases[j]);
        }

        printf("\nWeight Gradient:\n");
        for (int i = 0; i < input_size; i++)
        {
            printf("[%f] ", d_w[i]);
        }

        printf("\nBias Gradient:\n");
        for (int i = 0; i < output_size; i++)
        {
            printf("[%f] ", d_b[i]);
        }
    }

    // // Print the output
    // printf("Output: [%f]\n", outputs[0]);

    // // Compute the Loss

    // printf("Weight Gradient:\n");
    // for (int i = 0; i < input_size; i++)
    // {
    //     printf("[%f] ", d_w[i]);
    // }

    // printf("\nBias Gradient:\n");
    // for (int i = 0; i < output_size; i++)
    // {
    //     printf("[%f] ", d_b[i]);
    // }

    return 0;
}