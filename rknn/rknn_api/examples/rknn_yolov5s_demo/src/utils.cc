#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <iostream>
#include <sys/time.h>
#include <math.h>

#include "help.h"

using namespace std;


static inline uint64_t getTimeInUs()
{
    uint64_t time;
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    time = static_cast<uint64_t>(tv.tv_sec) * 1000000 + tv.tv_usec;
    return time;
}

template <typename T>
std::vector<size_t>
argsort_descend(const std::vector<T> &v)
{
    std::vector<size_t> indices(v.size());

    size_t n = 0;
    std::generate(indices.begin(), indices.end(), [&n] { return n++; });

    std::sort(indices.begin(), indices.end(),
              [&v](const size_t i, const size_t j) { return v[i] > v[j]; });

    return indices;
}

float compute_iou(
    const BoxCornerEncoding &lhs, const BoxCornerEncoding &rhs)
{
    float lhs_area = (lhs.ymax - lhs.ymin) * (lhs.xmax - lhs.xmin);
    float rhs_area = (rhs.ymax - rhs.ymin) * (rhs.xmax - rhs.xmin);

    if (lhs_area <= 0.0 || rhs_area <= 0.0)
    {
        return 0;
    }

    float intersection_ymin = std::max(lhs.ymin, rhs.ymin);
    float intersection_xmin = std::max(lhs.xmin, rhs.xmin);
    float intersection_ymax = std::min(lhs.ymax, rhs.ymax);
    float intersection_xmax = std::min(lhs.xmax, rhs.xmax);

    float dh = std::max(intersection_ymax - intersection_ymin, 0.0f);
    float dw = std::max(intersection_xmax - intersection_xmin, 0.0f);

    float intersection = dh * dw;

    float area_union = lhs_area + rhs_area - intersection;

    return intersection / area_union;
}

vector<size_t>
nms_single_class(
    const PredictBoxes &boxes,
    const PredictScores &scores,
    float confidence_threshold = 0.001,
    float iou_threshold = 0.5)
{
    //   std::cout << "=========" << std::endl;
    vector<size_t> desc_ind = argsort_descend(scores);
    vector<int> suppression;

    int last_elem = -1;
    for (int i = 0; i < desc_ind.size(); i++)
    {
        size_t idx = desc_ind[i];
        if (scores[idx] >= confidence_threshold)
        {
            last_elem = i;

            suppression.push_back(i);
        }
        else
        {
            break;
        }
    }

    vector<size_t> selected;
    for (int i = 0; i <= last_elem; i++)
    {
        if (suppression[i] == -99)
        {
            //      cout << "index " << i << " in score index array is already suppressed.\n";
            // ++i;
            continue;
        }

        size_t idx = desc_ind[i]; /* box index i */
        const BoxCornerEncoding &cur_box = boxes[idx];

        selected.emplace_back(idx);

        int j = i + 1;
        while (j <= last_elem)
        {
            size_t jdx = desc_ind[j]; /* box index j */
            const BoxCornerEncoding &box_j = boxes[jdx];

            float iou = compute_iou(cur_box, box_j);
            //   std::cout << "i:" << i << ",j:" << j << ",idx:" << idx << ",jdx:" << jdx << ",iou:" << iou << std::endl;
            assert(iou >= 0.0);
            /*
            * if iou is above threshold, then suppress box_j.
            * otherwise box_j will be the next *new* box.
            */
            if (iou >= iou_threshold && suppression[j] != -99)
            {
                suppression[j] = -99;
            }

            ++j;
        }

        //i = j;
    }

    return selected;
}

struct Det
{
    float c_x;
    float c_y;
    float b_w;
    float b_h;
    float score;
    int class_id;
};

//TODO
float clip(float val)
{
    float res;
    if (val > 1.0)
    {
        res = 1.0;
    }
    else
    {
        res = val;
    }

    if (val < 0.0)
    {
        res = 0.0;
    }
    else
    {
        res = val;
    }

    return res;
}
