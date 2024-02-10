#include <torch/torch.h>
#include "utils.h"
#include <opencv2/opencv.hpp>
#include "models.h"
#include "datasets.h"

int main()
{
    // Create a new Net.
    auto net = std::make_shared<Net>();

    // Create a multi-threaded data loader for the MNIST dataset.
    std::string path = "/home/yehoshua/projects/mnist_cpp/file_names.csv";
    auto csv = ReadCsv(path);
    auto training_dataset = CustomDataset(csv).map(torch::data::transforms::Stack<>());
    auto dbug_dataset = CustomDataset(csv);
    auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(training_dataset), /*batch_size=*/4);

    // // Instantiate an SGD optimization algorithm to update our Net's parameters.
    torch::optim::SGD optimizer(net->parameters(), /*lr=*/0.01);

    // TODO: finish dataloader and train the model save model push, create class and lib push code on github 1rst project.

    for (size_t epoch = 1; epoch <= 100; ++epoch)
    {
        size_t batch_index = 0;
        // Iterate the data loader to yield batches from the dataset.
        for (auto &batch : *data_loader)
        {
            // Reset gradients.
            optimizer.zero_grad();
            // Execute the model on the input data.
            torch::Tensor prediction = net->forward(batch.data);
            // Compute a loss value to judge the prediction of our model.
            torch::Tensor loss = torch::nll_loss(prediction, batch.target);
            // Compute gradients of the loss w.r.t. the parameters of our model.
            loss.backward();
            // Update the parameters based on the calculated gradients.
            optimizer.step();
            // Output the loss and checkpoint every 100 batches.
            if (++batch_index % 100 == 0)
            {
                std::cout << "Epoch: " << epoch << " | Batch: " << batch_index
                          << " | Loss: " << loss.item<float>() << std::endl;
                // Serialize your model periodically as a checkpoint.
                torch::save(net, "net.pt");
            }
        }
    }
}