add_library(ext-blaze INTERFACE)

target_include_directories(ext-blaze
	INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/ext/blaze/blaze"
	"${CMAKE_CURRENT_SOURCE_DIR}/ext/blaze"
)
