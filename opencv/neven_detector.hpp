//
//  neven_detector.hpp
//  sift_tracker
//
//  Created by Catalin Voss on 7/7/15.
//  Copyright (c) 2015 Sension, Inc. All rights reserved.
//

#ifndef __neven_detector__
#define __neven_detector__

#include <stdio.h>
#include <iostream>
#include <opencv2/core.hpp>
#ifdef ANDROID_NDK
#include "neven.h"
#else
#include <neven/neven.h>
#endif

/**
 * C++ wrapper for C neven class that allows us to easily go back to OpenCV land.
 */
class neven_detector {
public:
    void load(std::string path);
    bool detect(cv::Mat &img, std::vector<cv::RotatedRect> &faces);
    
private:
    struct neven_env *nev;
    int width;
    int height;
    std::string model;
};

#endif /* defined(__neven_detector__) */
