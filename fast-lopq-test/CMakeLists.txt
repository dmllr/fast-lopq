add_executable(fast-lopq-test
    main.cpp
    argsparser.cpp
    argsparser.h
)

target_link_libraries(fast-lopq-test
    fast-lopq
)
