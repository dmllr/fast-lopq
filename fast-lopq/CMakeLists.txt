set(generated-lopq_model "${CMAKE_CURRENT_BINARY_DIR}/lopq_model.pb.cc")

add_library(fast-lopq
    ${generated-lopq_model}

    model.cpp
    searcher.cpp

    include/fast-lopq/model.h
    include/fast-lopq/searcher.h
)

add_custom_command(
    OUTPUT ${generated-lopq_model}
    COMMAND protoc -I ../proto --cpp_out="${CMAKE_CURRENT_BINARY_DIR}" lopq_model.proto
    COMMENT "Run protoc to generate lopq model source in ${CMAKE_CURRENT_BINARY_DIR}"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(fast-lopq
    ext-blaze
    libprotobuf
)

target_include_directories(fast-lopq
    PUBLIC "include"
    PRIVATE "include/fast-lopq"
    PRIVATE "${CMAKE_CURRENT_BINARY_DIR}"
)
