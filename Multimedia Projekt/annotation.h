#pragma once

#include <string>
#include <vector>
#include <opencv2/core/core.hpp>	// Point, Rect, Vec3i

namespace mmp
{
	namespace annotation
	{
		struct element
		{
			unsigned id;
			std::string label;
			std::string original_label;
			cv::Point center;
			cv::Rect bounding_box;
		};

		class parse_error
		{
		private:
			std::string error_message;

		public:
			parse_error(const std::string& error_message)
				: error_message(error_message)
			{ }

			std::string error_msg() const { return error_message; }
			bool operator!() const { return error_message.empty(); }
		};

		class file
		{
		private:
			std::string image_filename;		// Image filename : "path/to/file.png"
			cv::Vec3i image_size;			// Image size (X x Y x C) : 123 x 456 x 3
			std::string database;			// Database : "src database"
			std::vector<element> objects;

		public:
			static parse_error parse(const std::string& filename, file& annotation);

			file();
			file(const std::string& img_filename, const cv::Vec3i img_size, const std::string& database = "");
			
			void add_object(const element& object);

			std::string get_image_filename() const { return image_filename; }
			cv::Vec3i get_image_size() const { return image_size; }
			std::string get_database() const { return database; }
			std::vector<element> get_objects() const { return objects; }
		};
	}
}