#include "device.cuh"

#include <cstdint>
#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>
#include <memory>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <fast-lopq/model.cuh>
#include <fast-lopq/searcher.cuh>


struct Searcher_ final : public lopq::gpu::Searcher {
	Searcher_(cublasHandle_t handle, const std::string& index_path) : lopq::gpu::Searcher(handle) {
		load_index_(index_path);
	}

	void load_index_(const std::string& index_path) {
		std::cout << " * loading index into memoory\n";

		std::string code_string;
		std::string id;

		std::ifstream raw_index(index_path);
		std::vector<lopq::gpu::Model::Codes> vectors;
		std::cout << "    - to RAM\n";
		while (raw_index >> code_string >> id) {
			lopq::gpu::Model::Codes fine_code(16);
			for (int i = 0; i < 16; ++i)
				sscanf(code_string.c_str() + 2 * i, "%2hhX", &fine_code[i]);

			cluster.ids.emplace_back(id);
			vectors.emplace_back(fine_code);
		}
		
		std::cout << "    - to device\n";
		cudaMalloc((void**)&cluster.vectors, 16 * vectors.size() * sizeof(uint8_t));
		uint32_t c = 0;
		for (auto v : vectors) {
			cublasSetVector(16, sizeof(uint8_t), &v[0], 1, &cluster.vectors[c * 16], 1);
			c++;
		}
	}

	lopq::gpu::Searcher::Cluster& get_cell(const lopq::gpu::Model::Codes& /*coarse_code*/) {
		return cluster;
	}

	lopq::gpu::Searcher::Cluster cluster;
};

void test_(const std::string& proto_path, const std::string& index_path) {
	cublasStatus_t stat;
	cublasHandle_t handle;

	stat = cublasCreate(&handle);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		printf("CUBLAS initialization failed\n");
		return;
	}
	printf("CUBLAS initialized\n");

	size_t sz = 128;
	scalar_t x[128] = {
			-5.15871673e-01,  2.96075179e-01, -1.29803211e-01,  7.89666891e-02,
			-2.13094767e-02,  1.22119331e-01,  8.73494489e-02,  6.42005949e-02,
			-3.91221325e-02,  6.10671771e-02,  5.30999659e-02, -3.33499452e-02,
 			 4.61418899e-02,  3.43788838e-02, -5.49158064e-02, -3.02026897e-02,
			 4.21869485e-02,  1.32635488e-02, -4.22456297e-02,  6.51733816e-02,
			 1.90685661e-02,  2.25571052e-02, -8.67056093e-02, -8.27722468e-02,
			 7.55270520e-02,  3.46607742e-02, -3.06546405e-02, -2.54700991e-02,
			 2.71249231e-02,  1.78375674e-02,  9.29383355e-02,  3.64048002e-02,
			-6.36318482e-02,  5.60158669e-03,  3.05002163e-03,  4.81614438e-02,
			 1.80568854e-02, -1.16311622e-02, -3.08939093e-02,  4.52400842e-02,
			 2.78512402e-02,  1.40238366e-03,  3.29228638e-02, -1.61220898e-02,
			 9.02076618e-05,  1.18825074e-02,  2.66880924e-02,  6.20281858e-02,
			-4.05081000e-02, -5.81048504e-03, -2.54299801e-03,  3.02054956e-02,
			 3.70625268e-02, -2.45962921e-02, -5.59338193e-02, -1.06997198e-02,
			-2.80183768e-02, -5.06158649e-02,  1.50465135e-02, -3.01377389e-02,
			 2.26933383e-03,  1.76983544e-02,  1.12419442e-02,  3.04122417e-03,
			 5.63636593e-03, -1.38508695e-01, -7.34715341e-02,  1.57006335e-01,
			 9.78414538e-02, -8.49019214e-02, -5.32735494e-02, -1.86980237e-01,
			-6.95972085e-02, -7.49042534e-02, -2.93728630e-02,  1.48451815e-01,
			-7.61983089e-02, -1.07924611e-01, -1.24433675e-01,  4.32059065e-02,
			 6.22441616e-02, -2.83085895e-02,  6.22212047e-02, -1.10452332e-01,
			-1.36765383e-04, -4.19165019e-02,  2.85524471e-02,  3.09949100e-02,
			-2.36794435e-02,  3.00126874e-02, -1.06457827e-02, -8.06230982e-03,
			 1.06088990e-01,  1.02251085e-02, -1.84198201e-02,  1.65926149e-04,
			 4.98833487e-02,  2.74232534e-02,  8.88374535e-03,  1.14191629e-03,
			-2.54754799e-02, -7.94487730e-03, -1.55596775e-02, -8.04370697e-03,
			 6.75116392e-03, -3.76623616e-03, -1.61214569e-02,  4.01648782e-03,
			-5.56433757e-02,  4.47427595e-03, -4.27675387e-02, -3.09779599e-02,
			 2.45707354e-02, -4.56435310e-02, -8.08994246e-04,  2.17876313e-02,
			 9.91254619e-03, -2.55167447e-02, -1.00904512e-02,  9.45845237e-03,
			 1.59078274e-02,  2.81953542e-03,  1.39462522e-02,  1.37137151e-03,
			-1.73925928e-02, -4.37456374e-03, -2.44480027e-02,  1.72845493e-03  };

	scalar_t* x_;
	cudaMalloc((void**)&x_, sz * sizeof(scalar_t));
	cublasSetVector(sz, sizeof(scalar_t), x, 1, x_, 1);

	Searcher_ searcher(handle, index_path);
	std::cout << " * loading model\n";
	searcher.load_model(proto_path);

	// printf(" * loading gpu model\n");
	// lopq::gpu::Model model(handle);
	// model.load(proto_path);

	auto coarse = searcher.model.predict_coarse(x_, sz);
	// std::cout << "   - predicted coarse codes: ";
	// for (uint8_t i = 0; i < 2; ++i)
	// 	std::cout << std::hex << (int)coarse[i] << std::dec << ' ';
	// std::cout << '\n';

	// auto fine = model.predict_fine(x_, sz, coarse);
	// std::cout << "   - predicted fine codes: ";
	// for (uint8_t i = 0; i < 16; ++i)
	// 	std::cout << std::hex << (int)fine[i] << std::dec << ' ';
	// std::cout << '\n';

	// auto sd = searcher.model.subquantizer_distances(x_, sz, coarse, 0);
	// std::cout << "    - subquantizer distances: " << sd.size << "x" << sd[0].size << "\n";
	// std::cout << "    : " << sd[0][0] << "\n";


	std::cout << "2. Testing of: LOPQ Searcher\n";


	std::cout << " * searching...\n";
	auto t0 = std::chrono::steady_clock::now();

	auto results = searcher.search(x_);

	auto t1 = std::chrono::steady_clock::now();
	std::cout << "    - got result in " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms\n";

	for (auto& r: results)
		std::cout << "      - " << r.id << "\n";

	// one_cell_of_index.reset();

	cublasDestroy(handle);
}
