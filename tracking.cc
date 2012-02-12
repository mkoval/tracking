#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

struct Target {
    std::vector<cv::Point> polygon_inner;
    std::vector<cv::Point> polygon_outer;
    cv::Mat mask;
    double score;
    bool dominant;
};

/* Target Dimensions
 * Black, Outside = 28" x 22"
 * White          = 24" x 18"
 * Black, Inside  = 14" x 20"
 */

static double const THRESHOLD = 100.0;
static double const THRESHOLD_AREA = 500.0;
static float const TARGET_WIDTH  = 24 * 0.0254; // meters
static float const TARGET_HEIGHT = 28 * 0.0254; // meters

// TODO: Actually calibrate the camera.
static cv::Mat const intrinsics = (cv::Mat_<float>(3, 3) <<
    510.157,   0.000, 354.999,
      0.000, 511.184, 226.924,
      0.000,   0.000,   1.000);

int main(int argc, char *argv[])
{
    if (argc <= 1) {
        std::cerr << "err: incorrect number of arguments\n"
                  << "usage: ./tracking <image>"
                  << std::endl;
        return 1;
    }

    std::string path = argv[1];
    cv::Mat color = cv::imread(path, 1);

    cv::Mat gray;
    cv::cvtColor(color, gray, CV_BGR2GRAY);

    cv::Mat edges;
    cv::Canny(gray, edges, 20, 200, 3);

    int const rows = color.rows;
    int const cols = color.cols;

    // Approximate contours as four-sided polygons.
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(edges, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

    std::vector<std::vector<cv::Point> > polygons;

    for (size_t i = 0; i < contours.size(); i++) {
        std::vector<cv::Point> contour = contours[i];
        double length = cv::arcLength(contour, true);

        std::vector<cv::Point> polygon;
        cv::approxPolyDP(contours[i], polygon, length * 0.02, true);
        polygons.push_back(polygon);
    }

    // Ignore the inner-most polygons and those without four vertices.
    std::vector<Target> targets;
    std::vector<double> scores;

    for (size_t i = 0; i < polygons.size(); i++) {
        std::vector<cv::Point> outer = polygons[i];
        if (outer.size() != 4) continue;

        // Search for first inner polygon with four corners.
        for (int j = hierarchy[i][2]; j > 0; j = hierarchy[j][1]) {
            std::vector<cv::Point> inner = polygons[j];
            if (inner.size() != 4) continue;

            // Mask the area between the inner and outer polygons.
            cv::Mat mask_inner(rows, cols, CV_8U, cv::Scalar(0));
            cv::Mat mask_outer(rows, cols, CV_8U, cv::Scalar(0));
            cv::fillConvexPoly(mask_inner, &inner[0], inner.size(), 255);
            cv::fillConvexPoly(mask_outer, &outer[0], outer.size(), 255);
            cv::Mat mask = mask_outer - mask_inner;

            // Compute the average brightness in the masked region.
            // TODO: Add a parallel line constraint.
            // TODO: Replace this with histogram matching to improve accuracy.
            cv::Scalar mean = cv::mean(color, mask);

            if (inner.size() == 4 && outer.size() == 4) {
                Target target;
                target.polygon_inner = inner;
                target.polygon_outer = outer;
                target.mask = mask;
                target.score = cv::norm(mean);
                target.dominant = true;
                targets.push_back(target);
            }
        }
    }

    // Check which targets intersect.
    // TODO: Make this more efficient by using a disjoint-sets data structure.
    std::vector<bool> dominant_flags(targets.size(), true);

    for (size_t i = 0; i < targets.size(); i++) {
        Target &ti = targets[i];

        for (size_t j = i + 1; j < targets.size(); j++) {
            Target &tj = targets[j];

            double max_overlap;
            cv::Mat overlap = cv::min(ti.mask, tj.mask);
            cv::minMaxLoc(overlap, NULL, &max_overlap);

            if (max_overlap) {
                if (ti.score > tj.score) {
                    tj.dominant = false;
                } else {
                    ti.dominant = false;
                }
            }
        }
    }

    // Filter out non-dominant targets.
    std::vector<Target> dominant_targets;

    for (size_t i = 0; i < targets.size(); i++) {
        double inner_area = cv::contourArea(targets[i].polygon_inner, false);
        double outer_area = cv::contourArea(targets[i].polygon_outer, false);
        double area = outer_area - inner_area;

        if (targets[i].dominant && area >= THRESHOLD_AREA) {
            dominant_targets.push_back(targets[i]);
        }
    }

    // Project the two-dimensional points back into the camera coordinate frame. Note
    // that the points are in counter-clockwise order starting from the top left.
    for (size_t i = 0; i < dominant_targets.size(); i++) {
        std::vector<cv::Point3f> polygon_3d(4);
        polygon_3d[0] = cv::Point3f(0.0f, 0.0f,         0.0f);
        polygon_3d[1] = cv::Point3f(0.0f, TARGET_WIDTH, 0.0f);
        polygon_3d[2] = cv::Point3f(0.0f, TARGET_WIDTH, TARGET_HEIGHT);
        polygon_3d[3] = cv::Point3f(0.0f, 0.0f,         TARGET_HEIGHT);

        std::vector<cv::Point2f> polygon_2d(4);
        for (size_t j = 0; j < dominant_targets[i].polygon_outer.size(); j++) {
            cv::Point2i pt_src = dominant_targets[i].polygon_outer[j];
            polygon_2d[j] = cv::Point2d(pt_src.x, pt_src.y);
        }

        cv::Vec3d tvec, rvec;
        cv::solvePnP(polygon_3d, polygon_2d, intrinsics, cv::Mat(), rvec, tvec, false);

        std::cout << "(" << tvec[0] << ", " << tvec[1] << ", " << tvec[2] << ")" << std::endl;
    }

    // Overlay the targets.
    for (size_t i = 0; i < dominant_targets.size(); i++) {
        Target target = dominant_targets[i];

        cv::Mat render;
        color.copyTo(render);

        std::vector<std::vector<cv::Point2i> > pts(2);
        pts[0] = target.polygon_inner;
        pts[1] = target.polygon_outer;
        cv::drawContours(render, pts, 0, cv::Scalar(255, 0, 0), 2);
        cv::drawContours(render, pts, 1, cv::Scalar(0, 0, 255), 2);

        for (size_t j = 0; j < 4; j++) {
            cv::circle(render, target.polygon_inner[j], 5, cv::Scalar(255, 0, 0), 1);
            cv::circle(render, target.polygon_outer[j], 5, cv::Scalar(0, 0, 255), 1);
        }
        std::cout << target.polygon_inner.size() << std::endl;

        cv::imshow("targets", render);
        while (cv::waitKey() != ' ');
    }
    return 0;
}
