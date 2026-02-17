// FILE: ./download_mnist.rs
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::Path;
use flate2::read::GzDecoder;
use reqwest::get;
use std::io::Read;

const MNIST_URLS: [(&str, &str); 4] = [
    ("train-images-idx3-ubyte.gz", "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz"),
    ("train-labels-idx1-ubyte.gz", "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz"),
    ("t10k-images-idx3-ubyte.gz", "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz"),
    ("t10k-labels-idx1-ubyte.gz", "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz"),
];

pub async fn download_mnist() -> io::Result<()> {
    let data_dir = "data";
    if !Path::new(data_dir).exists() {
        fs::create_dir(data_dir)?;
    }

    for (filename, url) in MNIST_URLS.iter() {
        let gz_path = format!("{}/{}", data_dir, filename);
        let out_path = gz_path.replace(".gz", "");

        if Path::new(&out_path).exists() {
            println!("  {} already exists", out_path);
            continue;
        }

        println!("  Downloading {}...", filename);
        let response = get(*url).await.map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        let bytes = response.bytes().await.map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

        println!("  Extracting {}...", filename);
        let mut gz = GzDecoder::new(&bytes[..]);
        let mut decompressed = Vec::new();
        gz.read_to_end(&mut decompressed)?;

        let mut file = File::create(&out_path)?;
        file.write_all(&decompressed)?;

        println!("  Saved: {}", out_path);
    }

    Ok(())
}
