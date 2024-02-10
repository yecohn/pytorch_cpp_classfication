#include "torch/torch.h"
#include <opencv2/opencv.hpp>

class CustomDataset : public torch::data::Dataset<CustomDataset>
{
private:
    std::vector<std::tuple<std::string, int64_t>> data;

public:
    CustomDataset(const std::vector<std::tuple<std::string, int64_t>> &csv_data)
        : data(csv_data) {}

    torch::data::Example<> get(size_t index) override;
    torch::optional<size_t> size() const override;
};