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

			class const_iterator
			{
			private:
				void * _word;

			private:
				friend class sparse_vector;
				const_iterator(void * _word);

			public:
				sparse_vector::size_type index() const;
				float operator*() const;
				const_iterator& operator++();
				bool operator==(const const_iterator& rhs) const;
				bool operator!=(const const_iterator& rhs) const { return !(*this == rhs); }
			};

		private:
			size_type _size;
			void * _vec;
			void * _word_end;

		public:
			sparse_vector(const sparse_vector& rhs);
			sparse_vector(sparse_vector&& rhs);

			sparse_vector(const cv::Mat& mat);
			~sparse_vector();

			float operator[](size_type i) const;
			size_type size() const { return _size; }
			void * data() { return _vec; }
			const void * data() const { return _vec; }

			const_iterator begin() const;
			const_iterator end() const;
		};

		std::string to_string(const sparse_vector& svec);

		class model
		{
		public:
			enum class example_type
			{
				negative = -1,
				positive = 1
			};

		private:
			void * _model;
			bool delete_svectors;
			std::vector<void *> docs;
			std::vector<double> targets;
			svm::sparse_vector::size_type vec_size;

		private:
			void push_back_example(const svm::sparse_vector& svec, example_type type);
			void learn();

		public:
			model& operator=(model&& rhs);
			model(model&& rhs);

			model() : _model(nullptr) { }
			model(const std::string& filename);

			template<class negatives_type, class positives_type>
			model(const negatives_type& negatives, const positives_type& positives, svm::sparse_vector::size_type vec_size)
				: vec_size(vec_size), delete_svectors(false)
			{
				docs.reserve(negatives.size() + positives.size());
				targets.reserve(negatives.size() + positives.size());

				for (auto& negative : negatives) { assert(negative.size() == vec_size); push_back_example(negative, example_type::negative); }
				for (auto& positive : positives) { assert(positive.size() == vec_size); push_back_example(positive, example_type::positive); }
				learn();
			}

			~model();

			double classify(const svm::sparse_vector& vec) const;
			void save(const std::string& filename) const;
		};
	}
}


