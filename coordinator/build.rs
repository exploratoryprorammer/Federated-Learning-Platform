fs main() {
    tonic_build::configure()
        .build_server(true)
        .compile(&["../proto/coordinator.proto"], & ["../proto"])
        .unwrap();
}