#include <cstdint>
#include <iostream>
#include <fstream>
#include <chrono>
#include <functional>

#include <blaze/Math.h>
#include <fast-lopq/model.h>
#include <fast-lopq/searcher.h>

#include "argsparser.h"

struct Searcher final : public lopq::Searcher {
	Searcher(Cluster& cluster)
		: cluster(cluster) {

	}

	Cluster& get_cell(const lopq::Model::CoarseCode& /*coarse_code*/) {
		return cluster;
	}

	Cluster cluster;
};

auto load_index(const std::string& index_path) {
	auto cluster = std::make_unique<lopq::Searcher::Cluster>();

	std::string code_string;
	std::string id;

	std::ifstream raw_index(index_path);
	while (raw_index >> code_string >> id) {

		lopq::Model::FineCode fine_code;
		for (int i = 0; i < 16; ++i)
			sscanf(code_string.c_str() + 2 * i, "%2hhX", &fine_code[i]);

		cluster->ids.emplace_back(id);
		cluster->vectors.emplace_back(fine_code);
		cluster->metadata.emplace_back(std::to_string(std::rand() % 10));
	}

	return cluster;
}

auto test(const std::function<std::vector<Searcher::Response>()>& runnable) {
	std::cout << " * searching...\n";
	auto t0 = std::chrono::steady_clock::now();

	auto results = runnable();

	auto t1 = std::chrono::steady_clock::now();
	std::cout << "    - got result in " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms\n";

	for (auto& r: results)
		std::cout << "      - " << r.id << ' ' << r.distance << '\n';
}


int main(int argc, char **argv) {
	ArgsParser args(argc, argv);

	const std::string &model_path = args.get("--proto-path");
	if (model_path.empty()) {
		std::cout << "--proto-path: No model file specified.";
		return EXIT_FAILURE;
	}
	const std::string &index_path = args.get("--index");
	if (index_path.empty()) {
		std::cout << "--index: No index filename specified.\n";
		return EXIT_FAILURE;
	}

	blaze::DynamicVector<double> x {
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
		-1.73925928e-02, -4.37456374e-03, -2.44480027e-02,  1.72845493e-03
	};


	std::cout << "1. Testing of: LOPQ Model\n";
	std::cout << " * loading model\n";
	lopq::Model model;
	model.load(model_path);

	auto coarse = model.predict_coarse(x);

	std::cout << "    - predicted coarse codes: ";
	for (uint8_t i = 0; i < coarse.size(); ++i)
		std::cout << std::hex << (int)coarse[i] << ' ';
	std::cout << '\n';

	auto fine = model.predict_fine(x, coarse);

	std::cout << "    - predicted fine codes: ";
	for (uint8_t i = 0; i < fine.size(); ++i)
		std::cout << std::hex << (int)fine[i] << ' ';
	std::cout << std::dec << '\n';

	std::cout << "2. Testing of: LOPQ Searcher\n";

	std::cout << " * loading index into memoory\n";
	auto one_cell_of_index = load_index(index_path);

	Searcher searcher(*one_cell_of_index);
	std::cout << " * loading model\n";
	searcher.load_model(model_path);

	test([&]() {
		std::cout << "   : .limit(13)\n";
		searcher
			.configure()
			.limit(13);

		return searcher.search(x);
	});

	test([&]() {
		std::cout << "   : .start(2).limit(13)\n";
		searcher
			.configure()
			.start(2)
			.limit(13);

		return searcher.search(x);
	});

	test([&]() {
		std::cout << "   : .limit(13).deduplication()\n";
		searcher
			.configure()
			.limit(13)
			.deduplication();

		return searcher.search(x);
	});

	test([&]() {
		std::cout << "   : .start(2).limit(13).deduplication()\n";
		searcher
			.configure()
			.start(2)
			.limit(13)
			.deduplication();

		return searcher.search(x);
	});

	test([&]() {
		std::cout << "   : limit(13).deduplication().filter(meta == '3')\n";
		searcher
			.configure()
			.limit(13)
			.deduplication()
			.filter([](auto& /*id*/, auto& meta) {
				return meta == "3";
			});

		return searcher.search(x);
	});

	test([&]() {
		std::cout << "   : limit(13).deduplication().filter(false)\n";
		searcher
			.configure()
			.limit(13)
			.deduplication()
			.filter([](auto& /*id*/, auto& /*meta*/) {
				return false;
			});

		return searcher.search(x);
	});

	one_cell_of_index.reset();

	std::cout << "DONE\n";

	return EXIT_SUCCESS;
}
