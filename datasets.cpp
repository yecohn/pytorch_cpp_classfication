#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include "datasets.h"

torch::data::Example<> CustomDataset::get(size_t index)
{

    std::string img_path = std::get<0>(data[index]);
    int label = std::get<1>(data[index]);

    // Load image with OpenCV
    cv::Mat img = cv::imread(img_path);
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);                // Convert to RGB
    cv::resize(img, img, cv::Size(28, 28), cv::INTER_LINEAR); // Resize image to 224x224

    // Convert the image and label to a tensor
    torch::Tensor img_tensor = torch::from_blob(img.data, {img.rows, img.cols, 1}, torch::kU8);
    img_tensor = img_tensor.permute({2, 0, 1});             // Convert to CxHxW
    img_tensor = img_tensor.toType(torch::kFloat).div(255); // Normalize to [0, 1]

    torch::Tensor label_tensor = torch::tensor(label, torch::kInt64);

    return {img_tensor.clone(), label_tensor};
}

torch::optional<size_t> CustomDataset::size() const
{
    return data.size();
}
