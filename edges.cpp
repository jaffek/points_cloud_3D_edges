#include <ogx/Plugins/EasyPlugin.h>
#include <ogx/Data/Clouds/CloudHelpers.h>
#include <ogx/Data/Clouds/KNNSearchKernel.h>
#include <ogx/Data/Clouds/SphericalSearchKernel.h>

using namespace ogx;
using namespace ogx::Data;

struct Find_edges_points : public ogx::Plugin::EasyMethod
{
	// fields
	Nodes::ITransTreeNode* m_node;

	// parameters
	Data::ResourceID m_node_id;
	Real sphere_radius;
	Real probability_thres;
	Integer clusters;

	// constructor
	Find_edges_points() : EasyMethod(L"Jaroslaw Affek",L"Finds cloud edges using weighted average criterion [Ruwen Schnabel]. Creates layer with probability values.")
	{
	}

	// add input/output parameters
	virtual void DefineParameters(ParameterBank& bank)
	{
		bank.Add(L"node_id", m_node_id).AsNode();
		bank.Add(L"Radius", sphere_radius = 0.6).Min(0.1).Max(2);
		bank.Add(L"Probability threshold", probability_thres = 0.9).Min(0).Max(1);
		bank.Add(L"Number of clusters", clusters = 4).Min(1).Max(10);
	}
	bool Init(Execution::Context& context)
	{
		OGX_SCOPE(log);
		// get node from id
		m_node = context.m_project->TransTreeFindNode(m_node_id);
		if (!m_node) ReportError(L"You must define node_id");

		OGX_LINE.Msg(User, L"Initialization succeeded");
		return EasyMethod::Init(context);
	}

	virtual void Run(Context& context)
	{
		auto subtree = context.Project().TransTreeFindNode(m_node_id);
		// report error if give node was not found, this will stop execution of algorithm
		if (!subtree) ReportError(L"Node not found");

		// update progress every 10000 points
		auto const progress_step = 10000;

		// run with number of threads available on current machine, optional
		auto const thread_count = std::thread::hardware_concurrency();

		// perform calculations for each cloud in given subtree
		Clouds::ForEachCloud(*subtree, [&](Clouds::ICloud & cloud, Nodes::ITransTreeNode & node)
		{
			// access points in the cloud
			Clouds::PointsRange points_all;
			cloud.GetAccess().GetAllPoints(points_all);

			// create vector for segmentation
			std::vector<StoredReal> segmentation_probabilities;

			// reserve points - create vector for probability values (segmentation values)
			std::vector<StoredReal> all_points_probabilities;
			all_points_probabilities.reserve(points_all.size());

			auto state_r = Clouds::RangeState(points_all);
			auto xyz_r = Data::Clouds::RangeLocalXYZ(points_all);
			auto state = state_r.begin();
			auto xyz = xyz_r.begin();
			int points_sum = 0;
			auto progress = 0;

			for (; xyz != xyz_r.end(); ++xyz, ++state)
			{
				// search for neighbors inside sphere
				Clouds::PointsRange neighbors;
				cloud.GetAccess().FindPoints(Clouds::SphericalSearchKernel(Math::Sphere3D(sphere_radius, xyz->cast<Real>())), neighbors);

				// get neighbors xyz parameters and sum number of neighbors
				Clouds::RangeLocalXYZConst neighbors_xyz(neighbors);
				auto neighbors_sum = neighbors.size();

				// fit a plane to the neighboring points and calculate projection of the current point on the plane
				auto const fitted_plane = Math::CalcBestPlane3D(neighbors_xyz.begin(), neighbors_xyz.end());
				Math::Point3D current_projxyz = Math::ProjectPointOntoPlane(fitted_plane, xyz->cast<Math::Point3D::Scalar>());

				// calculate and save store distances between neighbors points projected onto plane and current projected point
				Real max_distance = 0;
				Real distances_sum = 0;
				std::vector<Real> distances;
				for (auto & tested_point : neighbors_xyz)
				{
					Math::Point3D tested_projxyz = Math::ProjectPointOntoPlane(fitted_plane, tested_point.cast<Math::Point3D::Scalar>());
					Real distance_temp = Math::CalcPointToPointDistance3D(current_projxyz, tested_projxyz);
					distances.push_back(distance_temp);
					if(distance_temp > max_distance) 
						max_distance = distance_temp;
					distances_sum += distance_temp;
				}

				// calculate weighted average [according to theoretical formulas]
				auto distances_iterator = distances.begin();
				auto neighbors_xyz_iterator = neighbors_xyz.begin();
				std::vector<Real> weighted_distances;
				Real weighted_distances_sum = 0;
				Math::Point3D weighted_points_coord_sum = Math::Point3D::Zero();
				for (; distances_iterator != distances.end(); distances_iterator++, neighbors_xyz_iterator++)
				{
					Real single_weighted_distance = *distances_iterator*(1/(std::sqrt(2*3.141593)*max_distance))*std::exp(((-1)*(*distances_iterator)*(*distances_iterator))/(2*max_distance*max_distance));
					weighted_distances.push_back(single_weighted_distance);
					weighted_distances_sum += single_weighted_distance;
					weighted_points_coord_sum += single_weighted_distance*neighbors_xyz_iterator->cast<Real>();
				}

				Math::Point3D calculated_point_xyz = weighted_points_coord_sum / weighted_distances_sum;
				Math::Point3D calculated_point_projxyz = Math::ProjectPointOntoPlane(fitted_plane, calculated_point_xyz);

				// calculate probability, that current point is on the edge [according to theoretical formula]
				Real proj_distance = Math::CalcPointToPointDistance3D(current_projxyz, calculated_point_projxyz);
				Real mulitplier = distances_sum/(2*neighbors_sum);
				Real edge_point_probability_value = proj_distance*3*3.141593/(4*mulitplier);
				if(edge_point_probability_value > 1)
					edge_point_probability_value = 1;
				if (edge_point_probability_value < 0)
					edge_point_probability_value = 0;
				if (edge_point_probability_value > probability_thres)
				{
					points_sum++;
					state->set(Data::Clouds::PS_SELECTED);
				}

				// save probability for every point
				all_points_probabilities.push_back(edge_point_probability_value);

				// udpate progress every 10k points and check if we should continue
				++progress;
				if (!(progress % progress_step))
				{
					// progress is from 0 to 1
					if (!context.Feedback().Update(float(progress) / points_all.size()))
					{
						throw EasyException();
					}
				}
			}

			// segmentation
			Real cluster_val = 0;
			for (auto& iterator : all_points_probabilities)
			{
				cluster_val = std::floor(iterator*(clusters));
				segmentation_probabilities.push_back(StoredReal(cluster_val));
			}
			// create segmentation layer 
			auto v_sLayerName_seg = L"segmentation";
			Data::Layers::ILayer *layer_seg;
			auto layers_seg = cloud.FindLayers(v_sLayerName_seg);
			// check if layer exist
			if (!layers_seg.empty())
				layer_seg = layers_seg[0];
			else
				layer_seg = cloud.CreateLayer(v_sLayerName_seg, 0);
			points_all.SetLayerVals(segmentation_probabilities, *layer_seg); // saving layer to cloud


			// create probability layer 
			auto v_sLayerName = L"probability";
			Data::Layers::ILayer *layer;
			auto layers = cloud.FindLayers(v_sLayerName);
			// check if layer exist
			if (!layers.empty())
				layer = layers[0];
			else
				layer = cloud.CreateLayer(v_sLayerName, 0); 
			points_all.SetLayerVals(all_points_probabilities, *layer); // saving layer to cloud

			OGX_LINE.Format(Info, L"Number of edge points found: %d", points_sum);

		}, thread_count); // run with given number of threads, optional parameter, if not given will run in current thread
	}
};

OGX_EXPORT_METHOD(Find_edges_points)