add_library(fast-lopq
    "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/lopq_model.pb.cc"

    model.cpp
    searcher.cpp

    include/fast-lopq/model.h
    include/fast-lopq/searcher.h
)

add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/lopq_model.pb.cc"
    COMMAND mkdir -p "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}"
    COMMAND protoc -I ../proto --cpp_out="${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}" lopq_model.proto
    COMMENT "Run protoc to generate lopq model source in ${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(fast-lopq
    ext-blaze
    libprotobuf
)

target_include_directories(fast-lopq
    PUBLIC "include"
    PRIVATE "include/fast-lopq"
    PRIVATE "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}"
)
