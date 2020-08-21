/****************************************************************************
*
*    Copyright (c) 2017 - 2018 by Rockchip Corp.  All rights reserved.
*
*    The material in this file is confidential and contains trade secrets
*    of Rockchip Corporation. This is proprietary information owned by
*    Rockchip Corporation. No part of this work may be disclosed,
*    reproduced, copied, transmitted, or used in any way for any purpose,
*    without the express written permission of Rockchip Corporation.
*
*****************************************************************************/

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <sys/time.h>
// #include "timer.h"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "rknn_api.h"

#include <unistd.h>
#include <map>

using namespace std;
using namespace cv;

/*-------------------------------------------
                  Functions
-------------------------------------------*/
// int file_exists(char *filename)
// {
//     return (access(filename, 0) == 0);
// }

bool exists_test(const string & file_path) {
    return ( access(file_path.c_str(), F_OK ) != -1 );
}

static inline uint64_t getTimeInUs() {
    uint64_t time;
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    time = static_cast<uint64_t>(tv.tv_sec) * 1000000 + tv.tv_usec;
    return time;
}

static void printRKNNTensor(rknn_tensor_attr *attr) {
    printf("index=%d name=%s n_dims=%d dims=[%d %d %d %d] n_elems=%d size=%d fmt=%d type=%d qnt_type=%d fl=%d zp=%d scale=%f\n", 
            attr->index, attr->name, attr->n_dims, attr->dims[3], attr->dims[2], attr->dims[1], attr->dims[0], 
            attr->n_elems, attr->size, 0, attr->type, attr->qnt_type, attr->fl, attr->zp, attr->scale);
}

static unsigned char *load_model(const char *filename, int *model_size)
{
    FILE *fp = fopen(filename, "rb");
    if(fp == nullptr) {
        printf("fopen %s fail!\n", filename);
        return NULL;
    }
    fseek(fp, 0, SEEK_END);
    int model_len = ftell(fp);
    unsigned char *model = (unsigned char*)malloc(model_len);
    fseek(fp, 0, SEEK_SET);
    if(model_len != fread(model, 1, model_len, fp)) {
        printf("fread %s fail!\n", filename);
        free(model);
        return NULL;
    }
    *model_size = model_len;
    if(fp) {
        fclose(fp);
    }
    return model;
}


/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char** argv)
{
    map<int, int> sizeMap = {
        {1,5},
        {2,160},
        {3,5},
        {4,1},
        {5,7},
        {6,10},
        {7,10},
        {8,10},
        {9,10},
        {10,10},
        {11,10},
        {12,10},
        {13,10},
        {14,10},
        {15,10},
        {16,10},
        {17,10},
        {18,10},
        {19,10},
        {20,10},
        {21,10},
        {22,10},
        {23,10},
        {24,10},
        {25,10},
        {26,10},
        {27,10},
        {28,10},
        {29,10},
        {30,10},
        {31,10},
        {32,10},
        {33,10},
        {34,10},
        {35,10},
        {36,10},
        {37,10},
        {38,10},
        {39,10},
        {40,10},
        {41,10},
        {42,20},
        {43,20},
        {44,20},
        {45,20},
        {46,20},
        {47,20},
        {48,20},
        {49,20},
        {50,20},
        {51,20},
        {52,20},
        {53,20},
        {54,20},
        {55,20},
        {56,20},
        {57,20},
        {58,20},
        {59,20},
        {60,40},
        {61,40},
        {62,40},
        {63,40},
        {64,40},
        {65,40},
        {66,40},
        {67,40},
        {68,40},
        {69,40},
        {70,40},
        {71,40},
        {72,40},
        {73,40},
        {74,40},
        {75,40},
        {76,40},
        {77,40},
        {78,5},
        {79,5},
        {80,5},
        {81,5},
        {82,5},
        {83,5},
        {84,5},
        {85,5},
        {86,5},
        {87,80},
        {88,80},
        {89,80},
        {90,80},
        {91,80},
        {92,80},
        {93,80},
        {94,80},
        {95,80},
        {96,80},
        {97,6},
        {98,176},
        {99,6},
        {100,1},
        {101,7},
        {102,11},
        {103,11},
        {104,11},
        {105,11},
        {106,11},
        {107,11},
        {108,11},
        {109,11},
        {110,11},
        {111,11},
        {112,11},
        {113,11},
        {114,11},
        {115,11},
        {116,11},
        {117,11},
        {118,11},
        {119,11},
        {120,11},
        {121,11},
        {122,11},
        {123,11},
        {124,11},
        {125,11},
        {126,11},
        {127,11},
        {128,11},
        {129,11},
        {130,11},
        {131,11},
        {132,11},
        {133,11},
        {134,11},
        {135,11},
        {136,11},
        {137,11},
        {138,22},
        {139,22},
        {140,22},
        {141,22},
        {142,22},
        {143,22},
        {144,22},
        {145,22},
        {146,22},
        {147,22},
        {148,22},
        {149,22},
        {150,22},
        {151,22},
        {152,22},
        {153,22},
        {154,22},
        {155,22},
        {156,44},
        {157,44},
        {158,44},
        {159,44},
        {160,44},
        {161,44},
        {162,44},
        {163,44},
        {164,44},
        {165,44},
        {166,44},
        {167,44},
        {168,44},
        {169,44},
        {170,44},
        {171,44},
        {172,44},
        {173,44},
        {174,6},
        {175,6},
        {176,6},
        {177,6},
        {178,6},
        {179,6},
        {180,6},
        {181,6},
        {182,6},
        {183,88},
        {184,88},
        {185,88},
        {186,88},
        {187,88},
        {188,88},
        {189,88},
        {190,88},
        {191,88},
        {192,88},
        {193,6},
        {194,192},
        {195,6},
        {196,1},
        {197,7},
        {198,12},
        {199,12},
        {200,12},
        {201,12},
        {202,12},
        {203,12},
        {204,12},
        {205,12},
        {206,12},
        {207,12},
        {208,12},
        {209,12},
        {210,12},
        {211,12},
        {212,12},
        {213,12},
        {214,12},
        {215,12},
        {216,12},
        {217,12},
        {218,12},
        {219,12},
        {220,12},
        {221,12},
        {222,12},
        {223,12},
        {224,12},
        {225,12},
        {226,12},
        {227,12},
        {228,12},
        {229,12},
        {230,12},
        {231,12},
        {232,12},
        {233,12},
        {234,24},
        {235,24},
        {236,24},
        {237,24},
        {238,24},
        {239,24},
        {240,24},
        {241,24},
        {242,24},
        {243,24},
        {244,24},
        {245,24},
        {246,24},
        {247,24},
        {248,24},
        {249,24},
        {250,24},
        {251,24},
        {252,48},
        {253,48},
        {254,48},
        {255,48},
        {256,48},
        {257,48},
        {258,48},
        {259,48},
        {260,48},
        {261,48},
        {262,48},
        {263,48},
        {264,48},
        {265,48},
        {266,48},
        {267,48},
        {268,48},
        {269,48},
        {270,6},
        {271,6},
        {272,6},
        {273,6},
        {274,6},
        {275,6},
        {276,6},
        {277,6},
        {278,6},
        {279,96},
        {280,96},
        {281,96},
        {282,96},
        {283,96},
        {284,96},
        {285,96},
        {286,96},
        {287,96},
        {288,96},
        {289,7},
        {290,208},
        {291,7},
        {292,1},
        {293,7},
        {294,104},
        {295,104},
        {296,104},
        {297,104},
        {298,104},
        {299,104},
        {300,104},
        {301,104},
        {302,104},
        {303,104},
        {304,13},
        {305,13},
        {306,13},
        {307,13},
        {308,13},
        {309,13},
        {310,13},
        {311,13},
        {312,13},
        {313,13},
        {314,13},
        {315,13},
        {316,13},
        {317,13},
        {318,13},
        {319,13},
        {320,13},
        {321,13},
        {322,13},
        {323,13},
        {324,13},
        {325,13},
        {326,13},
        {327,13},
        {328,13},
        {329,13},
        {330,13},
        {331,13},
        {332,13},
        {333,13},
        {334,13},
        {335,13},
        {336,13},
        {337,13},
        {338,13},
        {339,13},
        {340,26},
        {341,26},
        {342,26},
        {343,26},
        {344,26},
        {345,26},
        {346,26},
        {347,26},
        {348,26},
        {349,26},
        {350,26},
        {351,26},
        {352,26},
        {353,26},
        {354,26},
        {355,26},
        {356,26},
        {357,26},
        {358,52},
        {359,52},
        {360,52},
        {361,52},
        {362,52},
        {363,52},
        {364,52},
        {365,52},
        {366,52},
        {367,52},
        {368,52},
        {369,52},
        {370,52},
        {371,52},
        {372,52},
        {373,52},
        {374,52},
        {375,52},
        {376,7},
        {377,7},
        {378,7},
        {379,7},
        {380,7},
        {381,7},
        {382,7},
        {383,7},
        {384,7},
        {385,7},
        {386,224},
        {387,7},
        {388,1},
        {389,7},
        {390,112},
        {391,112},
        {392,112},
        {393,112},
        {394,112},
        {395,112},
        {396,112},
        {397,112},
        {398,112},
        {399,112},
        {400,14},
        {401,14},
        {402,14},
        {403,14},
        {404,14},
        {405,14},
        {406,14},
        {407,14},
        {408,14},
        {409,14},
        {410,14},
        {411,14},
        {412,14},
        {413,14},
        {414,14},
        {415,14},
        {416,14},
        {417,14},
        {418,14},
        {419,14},
        {420,14},
        {421,14},
        {422,14},
        {423,14},
        {424,14},
        {425,14},
        {426,14},
        {427,14},
        {428,14},
        {429,14},
        {430,14},
        {431,14},
        {432,14},
        {433,14},
        {434,14},
        {435,14},
        {436,28},
        {437,28},
        {438,28},
        {439,28},
        {440,28},
        {441,28},
        {442,28},
        {443,28},
        {444,28},
        {445,28},
        {446,28},
        {447,28},
        {448,28},
        {449,28},
        {450,28},
        {451,28},
        {452,28},
        {453,28},
        {454,56},
        {455,56},
        {456,56},
        {457,56},
        {458,56},
        {459,56},
        {460,56},
        {461,56},
        {462,56},
        {463,56},
        {464,56},
        {465,56},
        {466,56},
        {467,56},
        {468,56},
        {469,56},
        {470,56},
        {471,56},
        {472,7},
        {473,7},
        {474,7},
        {475,7},
        {476,7},
        {477,7},
        {478,7},
        {479,7},
        {480,7},
    };
    ofstream outfile;
    outfile.open("rv1126_operation_cost.dat", ios::out | ios::trunc );
    const string postfixs[2] = {"test", "fake"};
    for (int idx = 1; idx <= 480; idx++) {
        for (string postfix : postfixs) {
            string filepath = "./rknn/" + to_string(idx) + "_" + postfix + ".rknn";
            
            if (exists_test(filepath)) {
                const int img_channels = 3;

                rknn_context ctx;
                int ret;
                int model_len = 0;
                unsigned char *model;

                // const char *model_path = "mobilenet_v2.rknn";
                const char *model_path = filepath.c_str();
                const char *img_path = "./dog_224x224.jpg";
                const int img_width = sizeMap[idx];
                const int img_height = sizeMap[idx];

                // Load image
                auto tic = getTimeInUs();
                cv::Mat orig_img = cv::imread(img_path, 1);
                cv::Mat img = orig_img.clone();
                if(!orig_img.data) {
                    printf("cv::imread %s fail!\n", img_path);
                    return -1;
                }
                if(orig_img.cols != img_width || orig_img.rows != img_height) {
                    // printf("resize %d %d to %d %d\n", orig_img.cols, orig_img.rows, img_width, img_height);
                    cv::resize(orig_img, img, cv::Size(img_width, img_height), (0, 0), (0, 0), cv::INTER_LINEAR);
                }
                // printf("Load image costs %8.3fms\n", (getTimeInUs() - tic) / 1000.0f); // 

                // Load RKNN Model
                tic = getTimeInUs();
                model = load_model(model_path, &model_len);
                ret = rknn_init(&ctx, model, model_len, 0);
                if(ret < 0) {
                    printf("rknn_init fail! ret=%d\n", ret);
                    return -1;
                }
                // printf("Load model costs %8.3fms\n", (getTimeInUs() - tic) / 1000.0f); // 

                // Get Model Input Output Info
                tic = getTimeInUs();
                rknn_input_output_num io_num;
                ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
                if (ret != RKNN_SUCC) {
                    printf("rknn_query fail! ret=%d\n", ret);
                    return -1;
                }
                // printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

                // printf("input tensors:\n");
                rknn_tensor_attr input_attrs[io_num.n_input];
                memset(input_attrs, 0, sizeof(input_attrs));
                for (int i = 0; i < io_num.n_input; i++) {
                    input_attrs[i].index = i;
                    ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
                    if (ret != RKNN_SUCC) {
                        // printf("rknn_query fail! ret=%d\n", ret);
                        return -1;
                    }
                    // printRKNNTensor(&(input_attrs[i]));
                }

                // printf("output tensors:\n");
                rknn_tensor_attr output_attrs[io_num.n_output];
                memset(output_attrs, 0, sizeof(output_attrs));
                for (int i = 0; i < io_num.n_output; i++) {
                    output_attrs[i].index = i;
                    ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
                    if (ret != RKNN_SUCC) {
                        printf("rknn_query fail! ret=%d\n", ret);
                        return -1;
                    }
                    // printRKNNTensor(&(output_attrs[i]));
                }
                // printf("Prepare costs %8.3fms\n", (getTimeInUs() - tic) / 1000.0f);
                // printf("input count is %d\n", io_num.n_input);

                const int loop_count = 100;
                float setinput_cost = 0.0;
                float inference_cost = 0.0;
                float getoutput_cost = 0.0;
                for (int i = 0; i < loop_count; i++){
                    // Set Input Data
                    tic = getTimeInUs();
                    // rknn_input inputs[1];
                    rknn_input inputs[io_num.n_input];
                    memset(inputs, 0, sizeof(inputs));
                    // printf("inputs count is %d", io_num.n_input);
                    for (uint32_t in_i = 0; in_i < io_num.n_input; in_i++) {
                        inputs[in_i].index = in_i;
                        inputs[in_i].type = RKNN_TENSOR_UINT8;
                        inputs[in_i].size = img.cols*img.rows*img.channels();
                        inputs[in_i].fmt = RKNN_TENSOR_NHWC;
                        inputs[in_i].buf = img.data;
                    }

                    ret = rknn_inputs_set(ctx, io_num.n_input, inputs);
                    if(ret < 0) {
                        // printf("rknn_input_set fail! ret=%d\n", ret);
                        return -1;
                    }
                    setinput_cost += (getTimeInUs() - tic) / 1000.0f;
                    // printf("set input data complete\t");

                    // Run
                    tic = getTimeInUs();
                    // printf("rknn_run\n");
                    ret = rknn_run(ctx, nullptr);
                    if(ret < 0) {
                        // printf("rknn_run fail! ret=%d\n", ret);
                        return -1;
                    }
                    inference_cost += (getTimeInUs() - tic) / 1000.0f;
                    // printf("inference complete\t");

                    // Get Output
                    tic = getTimeInUs();
                    rknn_output outputs[1];
                    memset(outputs, 0, sizeof(outputs));
                    outputs[0].want_float = 1;
                    ret = rknn_outputs_get(ctx, 1, outputs, NULL);
                    if(ret < 0) {
                        // printf("rknn_outputs_get fail! ret=%d\n", ret);
                        return -1;
                    }
                    getoutput_cost += (getTimeInUs() - tic) / 1000.0f;
                    // Release rknn_outputs
                    rknn_outputs_release(ctx, 1, outputs);
                    // printf("get output complete\n");
                }
                // printf("setinput_cost %8.3fms\n", setinput_cost / loop_count); // 
                std::cout << filepath << " : " << inference_cost / loop_count << endl;
                // char data[500];
                string data = to_string(idx) + "," + postfix + "," + to_string(inference_cost / loop_count);
                outfile << data.c_str() << endl;
                // printf("inference_cost %8.3fms\n", inference_cost / loop_count); // 
                // printf("getoutput_cost %8.3fms\n", getoutput_cost / loop_count); // 

                // Release
                if(ctx >= 0) {
                    rknn_destroy(ctx);
                }
                if(model) {
                    free(model);
                }
        }
        
        }
        // break;
    }
    outfile.close();

    
    return 0;
}
