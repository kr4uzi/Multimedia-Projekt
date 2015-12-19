#pragma once
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>	// Mat

namespace mmp
{
	namespace svm
	{
		class sparse_vector
		{
		public:
			typedef std::size_t size_type;

		private:
			size_type _size;
			void * _vec;

		public:
			sparse_vector(sparse_vector&& rhs);	// only move is allowed

			sparse_vector(const cv::Mat& mat);
			~sparse_vector();

			float operator[](size_type i) const;
			size_type size() const { return _size; }
			void * data() { return _vec; }
			const void * data() const { return _vec; }
		};

		class model
		{
		private:
			void * _model;
			std::vector<void *> docs;
			std::vector<double> targets;
			svm::sparse_vector::size_type vec_size;

		private:
			void push_back_example(const svm::sparse_vector& svec, char sign);

		public:
			model& operator=(model&& rhs);
			model(model&& rhs);

			model() : _model(nullptr) { }
			model(const std::string& filename);

			template<class negatives_type, class positives_type>
			model(const negatives_type& negatives, const positives_type& positives, svm::sparse_vector::size_type vec_size)
				: vec_size(vec_size)
			{
				docs.reserve(negatives.size() + positives.size());
				targets.reserve(negatives.size() + positives.size());

				for (auto& negative : negatives) push_back_example(negative, -1);
				for (auto& positive : positives) push_back_example(positive, 1);
			}

			~model();

			void learn();
			double classify(const svm::sparse_vector& vec) const;
			void save(const std::string& filename) const;
		};
	}
}


