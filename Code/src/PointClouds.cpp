#include "PointClouds.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#define DIMENSIONS 3

#define pointcloud 0
#define Colour_pointcloud 0
#define ERWTHMA_2 0
#define USE_KD_TREE 1
#define USE_SOBEL 1
#define TRIANGULATION 1
#define TOTAL_POINTCLOUD 1

using namespace std;
using namespace vvr;
using namespace Eigen;
using namespace cv;



PointCloudScene::PointCloudScene()
{
	//! Load settings.
	vvr::Shape::DEF_LINE_WIDTH = 4;
	vvr::Shape::DEF_POINT_SIZE = 6;
	m_perspective_proj = true;
	m_bg_col = Colour("768E77");
	m_point_color = Colour("454545");

	//m_KDree = NULL;

	for (int i = 1; i < count; ++i)
	{	//load image
		
		 //our file names are in the structure desk_1_1.png, desk_1_2.png,...
		char name_image[80];
		sprintf(name_image, "../resources/images/table_1/table_1_%d.png", i);
		images.push_back(cv::imread(name_image, CV_LOAD_IMAGE_ANYCOLOR));

		//load depth image
		cv::Mat depth_img;
		cv::Mat cnv_depth_img;

		char name_depth[80];
		sprintf(name_depth, "../resources/images/table_1/table_1_%d_depth.png", i);
		depth_img = cv::imread(name_depth, CV_LOAD_IMAGE_ANYDEPTH);

		double maxDepth;
		cv::minMaxLoc(depth_img, (double*)0, &maxDepth);
		depth_img.convertTo(cnv_depth_img, CV_8U, 255.0f / maxDepth);
		depths.push_back(cnv_depth_img);

	}

	/// GET POINTCLOUD
	getPointCloud(depths, m_pointclouds);
	float kd1;
	if (Colour_pointcloud) {

		m_colours.resize(images.size());

		for (int i = 0; i < images.size(); i++) {
			m_colours[i].resize(images[i].cols*images[i].rows);
			for (int k = 0; k < images[i].cols; k += 1) {
				for (int j = 0; j < images[i].rows; j += 1) {
					Vec3b intensity = images[i].at<Vec3b>(j, k);
					uchar red = intensity.val[0];
					uchar green = intensity.val[1];
					uchar blue = intensity.val[2];
					m_colours[i][j*images[i].cols / step + k / step] = (vec(red, green, blue));


				}
			}

		}
		echo(m_colours[num_pointcloud].size());
		echo(m_pointcloud_triang[num_pointcloud].size());
		echo(m_colours.size());
	}
	if (pointcloud == 0) {
		echo(m_colours.size());
		//echo(m_pointclouds.size());
		if (USE_SOBEL) {
			for (int i = 0; i < images.size(); i++) {
				Mat new_image;
				sobel(images[i], new_image);

				im_sobel.push_back(new_image);
			}
			//echo(im_sobel.size());

		//Mat src;
		//Sobel(src, sobelx, CV_32F, 1, 0);
			float thresholding = 20;
			getSobelPointCloud(im_sobel, m_sobel_pointclouds, thresholding);
		}

		///KD TREE
		if (ERWTHMA_2) {
			/// CLOSEST NEIGHBOR 
			float total_time;
			if (USE_KD_TREE) {

				int image = image_to;
				const float t = vvr::getSeconds();

				m_KDree = new KDree(m_pointclouds[image-1]);
				echo(m_pointclouds[image - 1].size());
				float dist;
				vec sc;
				for (int i = 0; i < m_pointclouds[image].size(); i++) {
					// gia kathe shmeio ths image-1 vriskw to kontinotero sthn image
					sc = vec(m_pointclouds[image][i].x, m_pointclouds[image][i].y, m_pointclouds[image][i].z);
					dist = std::numeric_limits<float>::max();
					const KDNode *nearest = NULL;

					KD_Nearest(sc, m_KDree->root(), nearest, dist);
					// to nn einai to kontinotero
					if (nearest) nn = nearest->split_point;

					m_close_points.push_back(m_pointclouds[image][i]); 
					m_close_points.push_back(vec(nn.x, nn.y, nn.z));
				}
				total_time = vvr::getSeconds() - t;
			}
			else {

				const float t = vvr::getSeconds();

				closest_neighbor(image_to, m_pointclouds, m_close_points);

				total_time = vvr::getSeconds() - t;
			}
			float x1, y1, z1, x2, y2, z2;
			float total_dist = 0;

			for (int points = 0; points < m_close_points.size() - 1; points = points + 2) {
				x1 = m_close_points[points].x;
				y1 = m_close_points[points].y;
				z1 = m_close_points[points].z;
				x2 = m_close_points[points + 1].x;
				y2 = m_close_points[points + 1].y;
				z2 = m_close_points[points + 1].z;
				total_dist = total_dist + sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2) + pow(z1 - z2, 2));
			}
			float ave_dist = total_dist / (m_close_points.size() / 2);
			echo(total_time);
			echo(total_dist);
			echo(ave_dist);


		}
		///FIND CLOSEST NEIGHBOR
		else {
			kd1 = vvr::getSeconds();

			if (USE_KD_TREE) {

				if (USE_SOBEL == 0) {
					for (int image = image_to; image > image_from; image--) {
						std::vector<vec> close_points;
						std::vector<vec> new_pointcloud1;
						std::vector<vec> new_pointcloud2;
						//ftiaxnw kd tree me to pointcloud ths image
						m_KDree = new KDree(m_pointclouds[image - 1]);

						float dist;
						vec sc;
						for (int i = 0; i < m_pointclouds[image].size(); i++) {
							// gia kathe shmeio ths image-1 vriskw to kontinotero sthn image
							sc = vec(m_pointclouds[image][i].x, m_pointclouds[image][i].y, m_pointclouds[image][i].z);
							dist = std::numeric_limits<float>::max();
							const KDNode *nearest = NULL;

							KD_Nearest(sc, m_KDree->root(), nearest, dist);
							// to nn einai to kontinotero
							if (nearest) nn = nearest->split_point;

							close_points.push_back(m_pointclouds[image][i]);
							close_points.push_back(vec(nn.x, nn.y, nn.z));
						}

						for (int j = 0; j < close_points.size() - 1; j = j + 2) {
							new_pointcloud1.push_back(close_points[j]);
							new_pointcloud2.push_back(close_points[j + 1]);
						}
						m_new.push_back(new_pointcloud1);
						m_new.push_back(new_pointcloud2);
						echo(image);

					}
				}
				else {

					for (int image = image_to; image > image_from; image--) {
						std::vector<vec> close_points;
						std::vector<vec> new_pointcloud1;
						std::vector<vec> new_pointcloud2;
						//ftiaxnw kd tree me to pointcloud ths image
						m_KDree = new KDree(m_sobel_pointclouds[image - 1]);

						float dist;
						vec sc;
						for (int i = 0; i < m_sobel_pointclouds[image].size(); i++) {
							// gia kathe shmeio ths image-1 vriskw to kontinotero sthn image
							sc = vec(m_sobel_pointclouds[image][i].x, m_sobel_pointclouds[image][i].y, m_sobel_pointclouds[image][i].z);
							dist = std::numeric_limits<float>::max();
							const KDNode *nearest = NULL;

							KD_Nearest(sc, m_KDree->root(), nearest, dist);
							// to nn einai to kontinotero
							if (nearest) nn = nearest->split_point;

							close_points.push_back(m_sobel_pointclouds[image][i]);
							close_points.push_back(vec(nn.x, nn.y, nn.z));
						}

						for (int j = 0; j < close_points.size() - 1; j = j + 2) {
							new_pointcloud1.push_back(close_points[j]);
							new_pointcloud2.push_back(close_points[j + 1]);
						}
						m_new.push_back(new_pointcloud1);
						m_new.push_back(new_pointcloud2);
						echo(image);

					}


				}
				
			}

			else {
				for (int i = image_to; i > image_from; i--) {
					std::vector<vec> close_points;
					std::vector<vec> new_pointcloud1;
					std::vector<vec> new_pointcloud2;
					//ME SOBEL XRHSIMOPOIEI TA SOBEL POINTCLOUDS MONO GIA TOYS KONTNOUTEROUS GEITONES
					// KAI GIA TO ROTATION METEPEITA PROFANWS
					if (USE_SOBEL) {
						closest_neighbor(i, m_sobel_pointclouds, close_points);
					}
					else {
						closest_neighbor(i, m_pointclouds, close_points);

					}
					for (int j = 0; j < close_points.size() ; j = j + 2) {
						new_pointcloud1.push_back(close_points[j]);
						new_pointcloud2.push_back(close_points[j + 1]);
					}
					m_new.push_back(new_pointcloud1);
					m_new.push_back(new_pointcloud2);
					echo(i);
				}
				float kd_total = -kd1 + vvr::getSeconds();
				echo(kd_total);

			}
		}

		if (TOTAL_POINTCLOUD) {
			/// CALCULATE ROTATION AND TRANSLATION
			// of the pointclouds
			for (int num_image = 1; num_image < m_new.size(); num_image = num_image + 2) {
				Eigen::MatrixXf t(3, 1);
				MatrixXf R;
				rotation_translation(num_image, m_new, R, t);

				all_rots.push_back(R);
				all_trans.push_back(t);
				std::cout << "ROTATION" << std::endl << std::endl << R << std::endl << std::endl;
				std::cout << "OPTIMAL TRANSLATION" << std::endl << std::endl << t << std::endl << std::endl;
			}
			echo(all_rots.size());
			//kanw pushback to teleutaio pointcloud pou den exei rot, trans
			m_rot_pointclouds.push_back(m_pointclouds[image_to]);

			/// IMLEMENT ROTATION AND TRANSLATION 
			// in each pointcloud
			int j = 0;

			for (int num = image_to - 1; num > image_from - 1; num--) {
				Eigen::MatrixXf total_R;
				echo(num);
				std::vector<vec> rotated;

				total_R = all_rots[0];
				for (int k = 1; k < j + 1; k++) {
					total_R = all_rots[k] * total_R;
				}

				for (int i = 0; i < m_pointclouds[num].size(); i++) {
					MatrixXf A(3, 1);
					A(0, 0) = m_pointclouds[num][i].x;
					A(1, 0) = m_pointclouds[num][i].y;
					A(2, 0) = m_pointclouds[num][i].z;

					MatrixXf final = total_R*A + all_trans[j];
					rotated.push_back(vec(final(0, 0), final(1, 0), final(2, 0)));
				}
				m_rot_pointclouds.push_back(rotated);
				/*
				for (int points = 0; points < m_rotated.size() - 1; points = points + 2) {
					x1 = m_rotated[points].x;
					y1 = m_rotated[points].y;
					z1 = m_rotated[points].z;
					x2 = m_pointclouds[image_to][points].x;
					y2 = m_pointclouds[image_to][points].y;
					z2 = m_pointclouds[image_to][points].z;
					total_dist = total_dist + sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2) + pow(z1 - z2, 2));
					x1 = m_pointclouds[image_to-1][points].x;
					y1 = m_pointclouds[image_to-1][points].y;
					z1 = m_pointclouds[image_to-1][points].z;
					total_dist2 = total_dist + sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2) + pow(z1 - z2, 2));

				}
				float ave_dist = total_dist / (m_close_points.size() / 2);
				echo(total_dist);
				echo(total_dist2);
				*/
			}
			float whole = vvr::getSeconds() - kd1;
			echo(whole);
			echo(m_rot_pointclouds.size());

			for (int i = 0; i < m_rot_pointclouds.size(); i++) {
				for (int j = 0; j < m_rot_pointclouds[i].size(); j++) {
					m_POINTCLOUD.push_back(m_rot_pointclouds[i][j]);
				}
			}
			

			/// DESTROY DUPLICATES
			/*
			float x2, y2, z2, y1, x1, z1;
			bool svisto;
			for (int i = 0; i < m_POINTCLOUD.size(); i++) {
				svisto = false;
				for (int j=0; j<no_duplicates.size(); j++){
					x1 = m_POINTCLOUD[i].x;
					x2 = no_duplicates[j].x;
					y1 = m_POINTCLOUD[i].y;
					y2 = no_duplicates[j].y;
					z1 = m_POINTCLOUD[i].z;
					z2 = no_duplicates[j].z;

					float dist = sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2) + pow(z1 - z2, 2));
					if (dist < 0.1) {
						svisto = true;
						break;
					}

				}
				if (svisto == false) {
					no_duplicates.push_back(m_POINTCLOUD[i]);
				}

			}
			echo(no_duplicates.size());
			*/
			/// USING KD-TREE
			float delet_duplicates_time = vvr::getSeconds() - whole;
			echo(delet_duplicates_time);
			/*
			echo(m_POINTCLOUD.size());
 
			m_KDree = new KDree(m_POINTCLOUD);
			float error = 0.3;
			float dist ;
			vec sc;
			for (int i = 0; i < m_POINTCLOUD.size(); i++) {
				// gia kathe shmeio tou POINTCLOUD vriskw to kontinotero tou sto POINTCLOUD
				sc = vec(m_POINTCLOUD[i].x, m_POINTCLOUD[i].y, m_POINTCLOUD[i].z);
				dist = std::numeric_limits<float>::max();
				const KDNode *nearest = NULL;

				KD_Nearest(sc, m_KDree->root(), nearest, dist);
				// to nn einai to kontinotero
				if (nearest) nn = nearest->split_point;
				//m_POINTCLOUD.erase()
				if (i % 1000 == 0) echo(i);
				if (dist > error) {
					no_duplicates.push_back(m_POINTCLOUD[i]);
				}

			}
			echo(no_duplicates.size());
			*/
			if (TRIANGULATION) {
				std::vector<float> distances;

				many_p(images[image_to], m_pointclouds[image_to], m_many_pclouds, distances);
				for (int i = 0; i < distances.size(); i++) {
					vvr::Mesh* model = nullptr;
					TriangulateMesh(m_pointcloud_triang[image_to], model, distances[i]);
					m_model.push_back(model);

				}
				int je = 0;
				for (int num = image_to - 1; num > image_from - 1; num--) {
					echo(num);
					std::vector<vec> rotated_triang;
					Eigen::MatrixXf total_R;

					total_R = all_rots[0];
					for (int k = 1; k < je + 1; k++) {
						total_R = all_rots[k] * total_R;
					}
					for (int i = 0; i < m_pointcloud_triang[num].size(); i++) {
						MatrixXf A(3, 1);
						A(0, 0) = m_pointcloud_triang[num][i].x;
						A(1, 0) = m_pointcloud_triang[num][i].y;
						A(2, 0) = m_pointcloud_triang[num][i].z;

						MatrixXf final = total_R * A + all_trans[je];
						rotated_triang.push_back(vec(final(0, 0), final(1, 0), final(2, 0)));
					}
					std::vector<float> distances;

					many_p(images[num], m_pointclouds[num], m_many_pclouds, distances);


					for (int i = 0; i < distances.size(); i++) {
						vvr::Mesh* model = nullptr;
						TriangulateMesh(rotated_triang, model, distances[i]);
						m_model.push_back(model);

					}
					je++;
				}

			}
		}
	}
	echo(m_POINTCLOUD.size());
	echo(m_model.size());
	reset();
}
void PointCloudScene::reset()
{
    Scene::reset();

	//delete m_KDree;
	//m_KDree = new KDree(m_pointclouds[1]);

    //! Define plane
    m_plane_d = 0;
    m_plane = Plane(vec(0, 1, 1).Normalized(), m_plane_d);

    //! Define what will be vissible by default
    m_style_flag = 0;
    m_style_flag |= FLAG_SHOW_SOLID;
    m_style_flag |= FLAG_SHOW_WIRE;
    m_style_flag |= FLAG_SHOW_AXES;
    m_style_flag |= FLAG_SHOW_POINTS;
    //m_style_flag |= FLAG_SHOW_PLANE;
}

void PointCloudScene::resize()
{
    //! By Making `first_pass` static and initializing it to true,
    //! we make sure that the if block will be executed only once.
    static bool first_pass = true;

    if (first_pass)
    {
		/*
        m_model_original.setBigSize(getSceneWidth() / 2);
        m_model_original.update();
        m_model = m_model_original;
		*/
		// Also get Point cloud from re-scaled model
		getPointCloud(depths,m_pointclouds);

		// Initialize OpenCV Windows for the images
		cv::namedWindow("image");

		cv::namedWindow("Depth Image n");
		cv::namedWindow("Depth Image n-1");
		if (USE_SOBEL) {
			cv::namedWindow("SOBEL");
		}


        first_pass = false;
    }
}


void PointCloudScene::arrowEvent(ArrowDir dir, int modif)
{
    math::vec n = m_plane.normal;
    if (dir == UP) m_plane_d += 1;
    if (dir == DOWN) m_plane_d -= 1;
    else if (dir == LEFT) n = math::float3x3::RotateY(DegToRad(1)).Transform(n);
    else if (dir == RIGHT) n = math::float3x3::RotateY(DegToRad(-1)).Transform(n);
    m_plane = Plane(n.Normalized(), m_plane_d);

}

void PointCloudScene::keyEvent(unsigned char key, bool up, int modif)
{
    Scene::keyEvent(key, up, modif);
    key = tolower(key);

    switch (key)
    {
    case 's': m_style_flag ^= FLAG_SHOW_SOLID; break;
    case 'w': m_style_flag ^= FLAG_SHOW_WIRE; break;
    case 'n': m_style_flag ^= FLAG_SHOW_NORMALS; break;
    case 'a': m_style_flag ^= FLAG_SHOW_AXES; break;
    case 'd': m_style_flag ^= FLAG_SHOW_DISTANCES; break;
    case 'b': m_style_flag ^= FLAG_SHOW_POINTS; break;
    }
}

void PointCloudScene::draw()
{
    //! Draw plane
	/*
    if (m_style_flag & FLAG_SHOW_SOLID) m_model.draw(m_point_color, SOLID);
    if (m_style_flag & FLAG_SHOW_WIRE) m_model.draw(Colour::black, WIRE);
    if (m_style_flag & FLAG_SHOW_NORMALS) m_model.draw(Colour::black, NORMALS);
    if (m_style_flag & FLAG_SHOW_AXES) m_model.draw(Colour::black, AXES);
	*/
	
	
	
	

	/*
	if (m_style_flag & FLAG_SHOW_POINTS) {
		for (int img = 0; img < m_rot_pointclouds.size(); img++) {
			vvr::Colour color(255, img*25,0);
			drawPointCloud(m_rot_pointclouds[img], color);
		}
	}
	
	for (int i = 0; i < m_many_pclouds.size(); i++) {
		vvr::Colour color(255, i * 5, 0);
		drawPointCloud(m_many_pclouds[i], color);
	}
	*/
	
	//if (m_style_flag & FLAG_SHOW_POINTS) {
	//	drawPointCloud(m_POINTCLOUD, vvr::Colour::red);

	//}
	
	/// DRAW LINE TO CLOSER POINT OF IMAGE N-1
	///2o ERWTHMA
	if (pointcloud) {
		if(Colour_pointcloud)
		drawColouredPointCloud(m_pointcloud_triang[num_pointcloud], m_colours[num_pointcloud]);
		else {
			drawPointCloud(m_pointclouds[num_pointcloud], Colour::red);

		}


	}
	else{
		if (ERWTHMA_2) {
			if (m_style_flag & FLAG_SHOW_POINTS) drawPointCloud(m_pointclouds[image_to - 1], Colour::red);
			if (m_style_flag & FLAG_SHOW_POINTS) drawPointCloud(m_pointclouds[image_to], Colour::blue);
			for (int i = 0; i < m_close_points.size(); i = i + 2) {
				LineSeg3D line(m_close_points[i].x, m_close_points[i].y, m_close_points[i].z, m_close_points[i + 1].x, m_close_points[i + 1].y, m_close_points[i + 1].z);
				line.draw();
			}
		}
		else if (TOTAL_POINTCLOUD) {
			if (TRIANGULATION) {
				for (int i = 0; i < m_model.size(); i++) {
					vvr::Colour color = vvr::Colour::red;
					if (m_style_flag & FLAG_SHOW_SOLID) m_model[i]->draw(color, SOLID);
					if (m_style_flag & FLAG_SHOW_WIRE) m_model[i]->draw(color, WIRE);
					if (m_style_flag & FLAG_SHOW_NORMALS) m_model[i]->draw(color, NORMALS);
					if (m_style_flag & FLAG_SHOW_AXES) m_model[i]->draw(color, AXES);
				}
			}
			else {
				if (m_style_flag & FLAG_SHOW_POINTS) {
					//drawPointCloud(m_rotated, vvr::Colour::red);
					//drawPointCloud(m_pointclouds[image_to], vvr::Colour::blue);
					//drawPointCloud(m_pointclouds[image_to-1], vvr::Colour::green);
					drawPointCloud(m_POINTCLOUD, vvr::Colour::red);
					//drawPointCloud(no_duplicates, vvr::Colour::red);

					//drawPointCloud(m_sobel_pointclouds[num_pointcloud], vvr::Colour::red);

				}
			}

		}
	}
	if (USE_SOBEL && TOTAL_POINTCLOUD == 0) {
		drawPointCloud(m_sobel_pointclouds[num_pointcloud], vvr::Colour::red);

	}
	// Display Images and update OpenCV
	// IMPORTANT: OpenCV finds open windows by name, e.g. "Color Image"
	//Point3D p1(p(0, 0), p(1, 0), p(2, 0), vvr::Colour::green);
	//Point3D q1(q(0, 0), q(1, 0), q(2, 0), vvr::Colour::green);
	//LineSeg3D linepq(p1.x, p1.y, p1.z, q1.x, q1.y, q1.z, vvr::Colour::black);
	//p1.draw();
	//q1.draw();
	//linepq.draw();
	cv::imshow("image", images[num_pointcloud]);

	cv::imshow("Depth Image n", depths[image_to]);
	cv::imshow("Depth Image n-1", depths[image_from]);
	if (USE_SOBEL) {
		cv::imshow("SOBEL", im_sobel[num_pointcloud]);
	}
	// Following is necessary to update CV's loop


	cv::waitKey(10);

}

int main(int argc, char* argv[])
{
    try {
        return vvr::mainLoop(argc, argv, new PointCloudScene);
    }
    catch (std::string exc) {
        cerr << exc << endl;
        return 1;
    }
    catch (...)
    {
        cerr << "Unknown exception" << endl;
        return 1;
    }
}

void PointCloudScene::getPointCloud(vector<cv::Mat> &depths, vector<vector<vec>> &m_pointclouds)
{
	const float fovWidth = DegToRad(48.6);
	const float fovHeight = DegToRad(62);
	echo(depths.size());
	for (int cnt = 0; cnt < depths.size()-1; cnt++) {
		std::vector<vec> current_cloud_1,current_cloud_2;
		current_cloud_2.resize(depths[1].cols*depths[1].rows);

		for (int i = 0; i < depths[cnt].cols; i += step) {
			for (int j = 0; j < depths[cnt].rows; j += step) {
				const float fx = depths[cnt].cols / (tanf(fovHeight / 2) * 2);
				const float fy = depths[cnt].rows / (tanf(fovWidth / 2) * 2);
				float Vo = depths[cnt].cols / 2;
				float Uo = depths[cnt].rows / 2;

				float Z = -depths[cnt].at<unsigned char>(j, i) / 10;
				float X = -Z * (i - Vo) / fx;
				float Y = Z  * (j - Uo) / fy;
				if (Z != 0) {
					current_cloud_1.push_back(vec(X, Y, Z));
					current_cloud_2[j*depths[1].cols/step + i/step] = (vec(X, Y, Z));
				}
			}
		}
		
		m_pointclouds.push_back(current_cloud_1);
		m_pointcloud_triang.push_back(current_cloud_2);

	}
}

void PointCloudScene::sobel(cv::Mat & image, cv::Mat & new_image)
{
	Mat src, src_gray;
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;

	src = image;
	GaussianBlur(src, src, Size(3, 3), 0, 0, BORDER_DEFAULT);
	cvtColor(src, src_gray, COLOR_BGR2GRAY);
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;
	//Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
	Sobel(src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
	//Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
	Sobel(src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);
	convertScaleAbs(grad_y, abs_grad_y);
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, new_image);
}


void PointCloudScene::getSobelPointCloud(std::vector<cv::Mat>& im_sobel, std::vector<std::vector<vec>>& sobel_pointclouds, float thresholding)
{

	const float fovWidth = DegToRad(48.6);
	const float fovHeight = DegToRad(62);
	
	for (int cnt = 0; cnt < im_sobel.size(); cnt++) {
		std::vector<vec> current_cloud;
		for (int i = 0; i < images[cnt].cols; i += step) {
			for (int j = 0; j < images[cnt].rows; j += step) {
				const float fx = depths[cnt].cols / (tanf(fovHeight / 2) * 2);
				const float fy = depths[cnt].rows / (tanf(fovWidth / 2) * 2);
				float Vo = depths[cnt].cols / 2;
				float Uo = depths[cnt].rows / 2;
				float Z = -depths[cnt].at<unsigned char>(j, i) / 10;
				float X = -Z * (i - Vo) / fx;
				float Y = Z  * (j - Uo) / fy;
				if (Z != 0) {
					float grey_value = im_sobel[cnt].at<unsigned char>(j, i);
					if (grey_value > thresholding) {
						current_cloud.push_back(vec(X, Y, Z));
					}
				}
			}
		}
		sobel_pointclouds.push_back(current_cloud);
	}
}
void PointCloudScene::drawPointCloud(vector<vec> m_vertices, const vvr::Colour& color)
{
	for (auto &d : m_vertices)
		Point3D(d.x, d.y, d.z, color).draw();
}

void PointCloudScene::drawColouredPointCloud(vector<vec> m_vertices, std::vector<vec> &colours)
{
	int j = 0;
	for (int i = 0; i < colours.size();i=i+step) {
		vvr::Colour color(colours[i].z, colours[i].y, colours[i].x);
		Point3D(m_vertices[i].x, m_vertices[i].y, m_vertices[i].z, color).draw();
		
	}
}

void PointCloudScene::closest_neighbor(int image, vector<vector<vec>> &m_pointclouds, vector<vec> &close_points) {

	float x1, y1, z1, x2, y2, z2, distance, index, min_dist;
	for (int i = 0; i < m_pointclouds[image].size(); i++) {
		x1 = m_pointclouds[image][i].x;
		y1 = m_pointclouds[image][i].y;
		z1 = m_pointclouds[image][i].z;
		min_dist = std::numeric_limits<float>::max();
		for (int j = 0; j < m_pointclouds[image-1].size(); j++) {
			x2 = m_pointclouds[image-1][j].x;
			y2 = m_pointclouds[image-1][j].y;
			z2 = m_pointclouds[image-1][j].z;

			distance = sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2) + pow(z1 - z2, 2));
			if (distance < min_dist) {
				min_dist = distance;
				//point i has minimum distance with j
				index = j;
			}
		}
		close_points.push_back(m_pointclouds[image][i]);
		close_points.push_back(m_pointclouds[image-1][index]);
	}
}

void::PointCloudScene::rotation_translation(int num_image, vector<vector<vec>> &m_new,  MatrixXf &R, MatrixXf &t) {
	

	std::cout << "size of image" << std::endl << std::endl << m_new[num_image].size() << std::endl << std::endl;
	std::cout << "size of image-1" << std::endl << std::endl << m_new[num_image - 1].size() << std::endl << std::endl;
	Eigen::MatrixXf p(3, 1);
	Eigen::MatrixXf q(3, 1);
	q(0, 0) = m_new[num_image - 1][0].x;
	q(1, 0) = m_new[num_image - 1][0].y;
	q(2, 0) = m_new[num_image - 1][0].z;
	p(0, 0) = m_new[num_image][0].x;
	p(1, 0) = m_new[num_image][0].y;
	p(2, 0) = m_new[num_image][0].z;

	int size = m_new[num_image].size();
	for (int i = 1; i < size; i++) {
		q(0, 0) = q(0, 0) + m_new[num_image - 1][i].x;
		q(1, 0) = q(1, 0) + m_new[num_image - 1][i].y;
		q(2, 0) = q(2, 0) + m_new[num_image - 1][i].z;

		p(0, 0) = p(0, 0) + m_new[num_image][i].x;
		p(1, 0) = p(1, 0) + m_new[num_image][i].y;
		p(2, 0) = p(2, 0) + m_new[num_image][i].z;
	}

	//Compute the centroids of both point sets

	q(0, 0) = q(0, 0) / size;
	q(1, 0) = q(1, 0) / size;
	q(2, 0) = q(2, 0) / size;

	p(0, 0) = p(0, 0) / size;
	p(1, 0) = p(1, 0) / size;
	p(2, 0) = p(2, 0) / size;


	std::cout << "Centroid p of current-1 image is:" << std::endl << std::endl << q << std::endl << std::endl;
	std::cout << "Centroid q of cuurent image is:" << std::endl << std::endl << p << std::endl << std::endl;

	//Compute the centered vectors
	Eigen::MatrixXf xi(3, size), yi(3, size);

	for (int i = 0; i < size; i++) {
		yi(0, i) = m_new[num_image - 1][i].x - q(0, 0);
		yi(1, i) = m_new[num_image - 1][i].y - q(1, 0);
		yi(2, i) = m_new[num_image - 1][i].z - q(2, 0);

		xi(0, i) = m_new[num_image][i].x - p(0, 0);
		xi(1, i) = m_new[num_image][i].y - p(1, 0);
		xi(2, i) = m_new[num_image][i].z - p(2, 0);
	}

	// COMPUTE ROTATION

	MatrixXf W;
	W.setOnes(size, 1);
	MatrixXf di = W.asDiagonal();
	MatrixXf S = xi  * di * yi.transpose();

	//NOW FIND SVD OF S
	Eigen::JacobiSVD<Eigen::MatrixXf> svd(S, Eigen::ComputeFullU | Eigen::ComputeFullV);
	//svd.matrixU() svd.matrixV()

	MatrixXf odet;
	odet.setZero(3, 3);
	odet(0, 0) = 1;
	odet(1, 1) = 1;
	odet(2, 2) = (svd.matrixV() * svd.matrixU().transpose()).determinant();

	R = svd.matrixV() * odet * svd.matrixU().transpose();

	//std::cout << "ROTATION" << std::endl << std::endl << R << std::endl << std::endl;

	//COMPUTE OPTIMAL TRANSLATION
	//Eigen::MatrixXf total_R;
	//total_R = all_rots[0];
	//for (int i = 1; i < all_rots.size(); i++) {
	//	total_R = all_rots[i] * total_R;
	//}
	t = q - R * p;

	//std::cout << "total_ROTATION" << std::endl << std::endl << total_R << std::endl << std::endl;

	//std::cout << "COMPUTE OPTIMAL TRANSLATION" << std::endl << std::endl << t << std::endl << std::endl;

}

void PointCloudScene::many_p(const cv::Mat& image, vector<vec> &vertices, vector<vector<vec>> &m_many_pclouds, std::vector<float> &distances) {

	std::vector<std::vector<vec>> dif_vert;
	float error = 0.2;
	int k = 0;
	std::vector<vec> clouds;
	std::vector<int> grammes;
	clouds.resize(vertices.size());
	grammes.resize(image.rows);
	for (int i = 0; i < 500; i++) {
		dif_vert.push_back(clouds);
	}
	//echo(vertices.size());
	distances.push_back(vertices[0].z);

	while (distances[0] < -1000) {
		k++;
		distances[0] = vertices[k].z;
	}

	for (int i = 0; i < vertices.size(); i++) {

		//if (i % trueDisp.rows == 0) {
		//	item[0].push_back(i);
		//}
		if (vertices[i].z > -1000) {

			//vector<float> dist_old = distances;
			for (int j = 0; j < distances.size(); j++) {

				if ((vertices[i].z > distances[j] - error) && (vertices[i].z < distances[j] + error)) {

					dif_vert[j][i] = vertices[i];

					break;
				}
				else if (j == distances.size() - 1) {
					distances.push_back(vertices[i].z);
					dif_vert[distances.size() - 1][i] = vertices[i];

				}
			}

		}
	}

	for (int i = 0; i < dif_vert.size(); i++) {
		std::vector<vec> current_cloud;
		for (int j = 0; j < dif_vert[i].size(); j++) {
			if (dif_vert[i][j].z > -1000) {
				current_cloud.push_back(dif_vert[i][j]);
			}
		}
		if (current_cloud.size() > 5) {
			m_many_pclouds.push_back(current_cloud);
		}
	}
}

void PointCloudScene::TriangulateMesh(const std::vector<vec>& vertices, vvr::Mesh*& mesh, float dist)
{
	//!//////////////////////////////////////////////////////////////////////////////////
	//! TASK:
	//!
	//!  - Create 2 triangles for every 4-vertex block
	//!  - Create mesh "m_model" (dont forget new Mesh())
	//!
	//!//////////////////////////////////////////////////////////////////////////////////

	mesh = new Mesh();
	vector<vec> myVertices;

	std::vector<vec>& modelVerts = mesh->getVertices();
	std::vector<vvr::Triangle>& tris = mesh->getTriangles();
	cv::Mat trueDisp = images[1];

	for (int k = 0; k < vertices.size(); k++) {
		int y = k / trueDisp.cols;
		int x = k % trueDisp.cols; // to upoloipo
		float error = 0.2;

		if ((x < trueDisp.cols - 1) && (y < trueDisp.rows - 1)) {
			if ((vertices[y*trueDisp.cols + x].z > dist - error) && (vertices[y*trueDisp.cols + x].z < dist + error)) {
				if ((vertices[y*trueDisp.cols + x].z > -1000) && (vertices[y*trueDisp.cols + x + 1].z > -1000) && (vertices[(y + 1)*trueDisp.cols + x].z > -100) && ((vertices[(y + 1)*trueDisp.cols + x + 1].z > -100))) {
				if ((vertices[y*trueDisp.cols + x].z > vertices[y*trueDisp.cols + x + 1].z - error) && (vertices[y*trueDisp.cols + x].z < vertices[y*trueDisp.cols + x + 1].z + error)) {
					if ((vertices[y*trueDisp.cols + x].z > vertices[(y + 1)*trueDisp.cols + x].z - error) && (vertices[y*trueDisp.cols + x].z < vertices[(y + 1)*trueDisp.cols + x].z + error)) {
						myVertices.push_back(vertices[y*trueDisp.cols + x]);
						myVertices.push_back(vertices[y*trueDisp.cols + x + 1]);
						myVertices.push_back(vertices[(y + 1)*trueDisp.cols + x]);


						if ((vertices[y*trueDisp.cols + x].z > vertices[(y + 1)*trueDisp.cols + x + 1].z - error) && (vertices[y*trueDisp.cols + x].z < vertices[(y + 1)*trueDisp.cols + x + 1].z + error)) {
							myVertices.push_back(vertices[y*trueDisp.cols + x + 1]);
							myVertices.push_back(vertices[(y + 1)*trueDisp.cols + x]);
							myVertices.push_back(vertices[(y + 1)*trueDisp.cols + x + 1]);

						}
					}
					if ((vertices[y*trueDisp.cols + x].z > vertices[(y + 1)*trueDisp.cols + x + 1].z - error) && (vertices[y*trueDisp.cols + x].z < vertices[(y + 1)*trueDisp.cols + x + 1].z + error)) {
						myVertices.push_back(vertices[y*trueDisp.cols + x]);
						myVertices.push_back(vertices[(y)*trueDisp.cols + x + 1]);
						myVertices.push_back(vertices[(y + 1)*trueDisp.cols + x + 1]);

					}
				}
			}
			}
		}
	}



	for (auto& d : myVertices) modelVerts.push_back(d);
	for (int i = 0; i < modelVerts.size(); i = i + 3) {
		tris.push_back(vvr::Triangle(&modelVerts, i, i + 1, i + 2));
	}

	//echo(tris.size());

}

KDree::KDree(VecVector &pts)
	: pts(pts)
{
	const float t = vvr::getSeconds();
	m_root = new KDNode();
	m_depth = makeNode(m_root, pts, 0);
	const float KDTree_construction_time = vvr::getSeconds() - t;
	echo(KDTree_construction_time);
	echo(m_depth);
}

KDree::~KDree()
{
	const float t = vvr::getSeconds();
	delete m_root;
	const float KDTree_destruction_time = vvr::getSeconds() - t;
	echo(KDTree_destruction_time);
}

int KDree::makeNode(KDNode *node, VecVector &pts, const int level)
{
	//! Sort along the appropriate axis, find median point and split.
	const int axis = level % DIMENSIONS;
	std::sort(pts.begin(), pts.end(), VecComparator(axis));
	const int i_median = pts.size() / 2;

	//! Set node members
	node->level = level;
	node->axis = axis;
	node->split_point = pts[i_median];
	node->aabb.SetFrom(&pts[0], pts.size());

	//! Continue recursively or stop.
	if (pts.size() <= 1)
	{
		return level;
	}
	else
	{
		int level_left = 0;
		int level_right = 0;

		VecVector pts_left(pts.begin(), pts.begin() + i_median);
		VecVector pts_right(pts.begin() + i_median + 1, pts.end());

		if (!pts_left.empty())
		{
			node->child_left = new KDNode();
			level_left = makeNode(node->child_left, pts_left, level + 1);

		}
		if (!pts_right.empty())
		{
			node->child_right = new KDNode();
			level_right = makeNode(node->child_right, pts_right, level + 1);
		}

		int max_level = std::max(level_left, level_right);
		return max_level;
	}
}

void KDree::getNodesOfLevel(KDNode *node, std::vector<KDNode*> &nodes, int level)
{
	if (!level)
	{
		nodes.push_back(node);
	}
	else
	{
		if (node->child_left) getNodesOfLevel(node->child_left, nodes, level - 1);
		if (node->child_right) getNodesOfLevel(node->child_right, nodes, level - 1);
	}
}

std::vector<KDNode*> KDree::getNodesOfLevel(const int level)
{
	std::vector<KDNode*> nodes;
	if (!m_root) return nodes;
	getNodesOfLevel(m_root, nodes, level);
	return nodes;
}


void KD_Nearest(const vec& test_pt, const KDNode* root, const KDNode* &nn, float& best_dist)
{
	// Find the Nearest Neighbour to "test_pt" and store it to "nn"
	// Traverse the tree from "root" down to search the best distance to "test_pt"
	// and then traverse the tree bottom up to search for alternatives,
	// store the best distance to "best_dist"
	// ...

	int axis = root->axis;

	//float d = sqrt(pow(test_pt.x - root->split_point.x, 2) + pow(test_pt.y - root->split_point.y, 2) + pow(test_pt.z - root->split_point.z, 2));
	float d = test_pt.Distance(root->split_point);
	if (d < best_dist) {
		best_dist = d;
		nn = root;
	}

	float distance = test_pt.ptr()[axis] - root->split_point.ptr()[axis];

	if (distance < 0) {
		if (root->child_left) {
			KD_Nearest(test_pt, root->child_left, nn, best_dist);
		}

		if (root->child_right) {
			KD_Nearest(test_pt, root->child_right, nn, best_dist);
		}
	}

	else {
		if (root->child_left) {
			KD_Nearest(test_pt, root->child_left, nn, best_dist);
		}

		if (root->child_right) {
			KD_Nearest(test_pt, root->child_right, nn, best_dist);
		}
	}

	if (best_dist > distance*distance) {

		if (distance < 0) {
			if (root->child_right) {
				KD_Nearest(test_pt, root->child_right, nn, best_dist);
			}
		}
		else {

			if (root->child_left) {
				KD_Nearest(test_pt, root->child_left, nn, best_dist);
			}
		}

	}
}
