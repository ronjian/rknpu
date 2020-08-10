#include "utils.cc"
#include <gtest/gtest.h>
#include <fstream>
#include <iostream>
#include <regex>
#include "string.h"

using namespace std;

static void split(const std::string &source, std::vector<std::string> &tokens, const string &delimiter = " ")
{
    regex re(delimiter);
    copy(sregex_token_iterator(source.begin(), source.end(), re, -1),
         sregex_token_iterator(),
         back_inserter(tokens));
}


TEST(NMSTest, IOU05THRES01)
{
    const int testCnt = 2407;
    for (int i = 1; i <= testCnt; i++)
    {
        std::cout << "testing idx: " << to_string(i) << std::endl;
        char buffer[256];
        PredictScores pred_scores;
        ifstream pred_scores_file;
        pred_scores_file.open("./nms-test-data/pred_scores_" + to_string(i) + ".dat", ios::in);
        while (!pred_scores_file.eof())
        {
            pred_scores_file.getline(buffer, 256);
            string s_buf = buffer;
            if (!s_buf.empty())
            {
                float score = atof(buffer);
                pred_scores.push_back(score);
            }
        }
        pred_scores_file.close();

        std::vector<std::size_t> selected;
        ifstream selected_file;
        selected_file.open("./nms-test-data/selected_" + to_string(i) + ".dat", ios::in);
        while (!selected_file.eof())
        {
            selected_file.getline(buffer, 256);
            string s_buf = buffer;
            if (!s_buf.empty())
            {
                size_t sel = atoi(buffer);
                selected.push_back(sel);
            }
        }
        selected_file.close();

        PredictBoxes pred_boxes;
        ifstream pred_boxes_file;
        pred_boxes_file.open("./nms-test-data/pred_boxes_" + to_string(i) + ".dat", ios::in);
        float xmin, ymin, xmax, ymax;
        while (!pred_boxes_file.eof())
        {
            pred_boxes_file.getline(buffer, 256);
            string str_buf = buffer;
            if (!str_buf.empty())
            {
                vector<std::string> tokens;
                split(buffer, tokens, ",");
                assert(tokens.size() == 4);
                BoxCornerEncoding box;
                box.xmin = atof(tokens[0].c_str());
                box.ymin = atof(tokens[1].c_str());
                box.xmax = atof(tokens[2].c_str());
                box.ymax = atof(tokens[3].c_str());
                pred_boxes.push_back(box);
            }
        }

        ASSERT_EQ(pred_boxes.size(), pred_scores.size());
        EXPECT_EQ(selected, nms_single_class(pred_boxes, pred_scores, 0.0f, 0.5f));
        // ASSERT_EQ (selected, nms_single_class(pred_boxes, pred_scores, 0.0f, 0.5f));
    }
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}