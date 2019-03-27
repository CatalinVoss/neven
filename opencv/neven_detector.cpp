//
//  neven_detector.cpp
//  sift_tracker
//
//  Created by Catalin Voss on 7/7/15.
//  Copyright (c) 2015 Sension, Inc. All rights reserved.
//

#include "neven_detector.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#define dbl at<double>
#define ucr at<uchar>
#define flt at<float>

using namespace std;
using namespace cv;

#pragma mark -
#pragma mark Constants

const static int kNMaxFaces = 3; // seems like a decent target? see how it impacts performance
const static int kRotations[] = {0, -15, 15, -20, 20, -30, 30};
const static int kNRotations = 6;
const static float kConfidenceThresh = 0.4;

#pragma mark -
#pragma mark Load

void neven_detector::load(string path) {
    model = path;
    // don't actually create the neven environment until we track the first face
    nev = NULL;
}

#pragma mark -
#pragma mark Detect

inline void rotate(cv::Mat &src, double angle, cv::Mat &dst, cv::Mat &rot) {
    if (angle == 0)
    { dst = src; }
    
    int len = std::max(src.cols, src.rows);
    Point2f pt(len/2., len/2.);
    rot = getRotationMatrix2D(pt, angle, 1.0);
    warpAffine(src, dst, rot, cv::Size(len, len), WARP_INVERSE_MAP);
}

bool neven_detector::detect(cv::Mat &img, std::vector<cv::RotatedRect> &faces) {
    // Neven doesn't seem to like images where w < h, so pad the img
    Mat padded;
    int pad = 0;
    if (img.cols < img.rows) {
        int padding_required = img.rows-img.cols;
        // make sure padding is even
        if (padding_required % 2 == 1) {
            padding_required += 1;
        }
        // Padding on each side
        pad = padding_required/2;
        // Create padded image
        copyMakeBorder(img, padded, 0, 0, pad, pad, cv::BORDER_CONSTANT, 155);
    } else {
        // here, pad == 0
        padded = img;
    }
    
    // (Re-)create neven module if we don't have one or the image size suddenly changed on us
    if (nev == NULL || width != padded.cols || height != padded.rows) {
        width = padded.cols;
        height = padded.rows;
        nev = neven_create(model.c_str(), width, height, kNMaxFaces);
    }
    
    // TODO CV: This can be easily parallelized, though we don't want to do more work per frame than required
    static Mat rotated;
    static Mat rotation_matrix;
    static Mat drawable;
    
    for (int i = 0; i < kNRotations; i++) {
        int rot = kRotations[i];
        rotate(padded, rot, rotated, rotation_matrix);
        int n = neven_detect(nev, rotated.data);
        
        if (n > 0) {
            for (int j = 0; j < n; j++) {
                struct neven_face face;
                neven_get_face(nev, &face, j);
                
                // Ignore faces below the confidence threshold
                if (face.confidence < kConfidenceThresh)
                { continue; }
                
                // Invert per rotation
                Mat r = (Mat_<double>(3,1) << face.midpointx, face.midpointy, 1);
                Mat p = rotation_matrix*r; // this works because we use WARP_INVERSE_MAP
                Point c(p.dbl(0), p.dbl(1));
                // Approximate rotated rect (where we assume that the face is decently close to the center, so the angle is approximately equal to the one we originally rotated with...)
                RotatedRect rotated(Point(p.dbl(0), p.dbl(1)+0.35*face.eyedist), Size(2.5*face.eyedist, 2.5*face.eyedist), -rot);
                rotated.center.x -= pad; // subtract out pad (if non-zero)
                // Store
                faces.push_back(rotated);
            }
            
            // Don't rotate any further to look for more stuff
            // We could change this to look for further faces, in particular if kNMaxFaces > 1
            return true;
        }
    }
    
    return false;
}


