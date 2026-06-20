// Parser minimo de metadados GGUF + scanner de pastas.
// Le apenas o header (magic, versao, contadores) e os pares chave/valor
// de metadados — o suficiente para arquitetura, n.o de camadas, contexto, etc.
// Nao carrega os tensores.

use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF" em little-endian

// Tipos de valor GGUF
const T_UINT8: u32 = 0;
const T_INT8: u32 = 1;
const T_UINT16: u32 = 2;
const T_INT16: u32 = 3;
const T_UINT32: u32 = 4;
const T_INT32: u32 = 5;
const T_FLOAT32: u32 = 6;
const T_BOOL: u32 = 7;
const T_STRING: u32 = 8;
const T_ARRAY: u32 = 9;
const T_UINT64: u32 = 10;
const T_INT64: u32 = 11;
const T_FLOAT64: u32 = 12;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub path: String,
    pub name: String,
    pub folder: String,
    pub arch: String,
    pub quant: String,
    pub size_bytes: u64,
    pub size_gb: f64,
    pub block_count: Option<u32>,
    pub context_length: Option<u32>,
    pub embedding_length: Option<u32>,
    pub head_count: Option<u32>,
    pub head_count_kv: Option<u32>,
    pub size_label: Option<String>,
    pub has_mmproj: bool,
    pub mmproj_path: Option<String>,
}

struct Reader<R: Read> {
    inner: R,
}

impl<R: Read> Reader<R> {
    fn new(inner: R) -> Self {
        Reader { inner }
    }
    fn read_exact_n(&mut self, n: usize) -> std::io::Result<Vec<u8>> {
        let mut buf = vec![0u8; n];
        self.inner.read_exact(&mut buf)?;
        Ok(buf)
    }
    fn u32(&mut self) -> std::io::Result<u32> {
        let b = self.read_exact_n(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn u64(&mut self) -> std::io::Result<u64> {
        let b = self.read_exact_n(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }
    fn string(&mut self) -> std::io::Result<String> {
        let len = self.u64()? as usize;
        // protege contra strings absurdas/corrompidas
        if len > 64 * 1024 * 1024 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "string GGUF grande demais",
            ));
        }
        let bytes = self.read_exact_n(len)?;
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }
}

// Valor parseado que nos interessa (numerico ou string).
enum Val {
    U(u64),
    I(i64),
    F(f64),
    S(String),
    Other,
}

fn type_size(t: u32) -> Option<usize> {
    Some(match t {
        T_UINT8 | T_INT8 | T_BOOL => 1,
        T_UINT16 | T_INT16 => 2,
        T_UINT32 | T_INT32 | T_FLOAT32 => 4,
        T_UINT64 | T_INT64 | T_FLOAT64 => 8,
        _ => return None,
    })
}

fn read_value<R: Read>(r: &mut Reader<R>, vtype: u32) -> std::io::Result<Val> {
    Ok(match vtype {
        T_UINT8 => Val::U(r.read_exact_n(1)?[0] as u64),
        T_INT8 => Val::I(r.read_exact_n(1)?[0] as i8 as i64),
        T_UINT16 => {
            let b = r.read_exact_n(2)?;
            Val::U(u16::from_le_bytes([b[0], b[1]]) as u64)
        }
        T_INT16 => {
            let b = r.read_exact_n(2)?;
            Val::I(i16::from_le_bytes([b[0], b[1]]) as i64)
        }
        T_UINT32 => Val::U(r.u32()? as u64),
        T_INT32 => {
            let b = r.read_exact_n(4)?;
            Val::I(i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as i64)
        }
        T_FLOAT32 => {
            let b = r.read_exact_n(4)?;
            Val::F(f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64)
        }
        T_BOOL => Val::U(r.read_exact_n(1)?[0] as u64),
        T_STRING => Val::S(r.string()?),
        T_UINT64 => Val::U(r.u64()?),
        T_INT64 => {
            let b = r.read_exact_n(8)?;
            Val::I(i64::from_le_bytes([
                b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
            ]))
        }
        T_FLOAT64 => {
            let b = r.read_exact_n(8)?;
            Val::F(f64::from_le_bytes([
                b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
            ]))
        }
        T_ARRAY => {
            let elem_type = r.u32()?;
            let count = r.u64()? as usize;
            // Pula o conteudo do array de forma eficiente quando possivel.
            if let Some(sz) = type_size(elem_type) {
                // tipos de tamanho fixo: pula tudo de uma vez
                let total = sz.saturating_mul(count);
                let mut remaining = total;
                let mut scratch = [0u8; 8192];
                while remaining > 0 {
                    let chunk = remaining.min(scratch.len());
                    r.inner.read_exact(&mut scratch[..chunk])?;
                    remaining -= chunk;
                }
            } else if elem_type == T_STRING {
                for _ in 0..count {
                    let _ = r.string()?;
                }
            } else if elem_type == T_ARRAY {
                // arrays aninhados: raros; aborta o parse de metadados aqui
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "array aninhado nao suportado",
                ));
            }
            Val::Other
        }
        _ => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "tipo GGUF desconhecido",
            ));
        }
    })
}

fn as_u32(v: &Val) -> Option<u32> {
    match v {
        Val::U(u) => Some(*u as u32),
        Val::I(i) => Some(*i as u32),
        _ => None,
    }
}

/// Le os metadados relevantes de um arquivo GGUF.
pub fn parse_header(path: &Path) -> std::io::Result<ParsedGguf> {
    let file = File::open(path)?;
    let mut r = Reader::new(BufReader::with_capacity(1 << 16, file));

    let magic = r.u32()?;
    if magic != GGUF_MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "magic GGUF invalido",
        ));
    }
    let _version = r.u32()?;
    let _tensor_count = r.u64()?;
    let kv_count = r.u64()?;

    let mut arch: Option<String> = None;
    let mut block_count: Option<u32> = None;
    let mut context_length: Option<u32> = None;
    let mut embedding_length: Option<u32> = None;
    let mut head_count: Option<u32> = None;
    let mut head_count_kv: Option<u32> = None;
    let mut size_label: Option<String> = None;
    let mut file_type: Option<u32> = None;
    let mut name: Option<String> = None;

    for _ in 0..kv_count {
        let key = r.string()?;
        let vtype = r.u32()?;
        let val = read_value(&mut r, vtype)?;

        match key.as_str() {
            "general.architecture" => {
                if let Val::S(s) = &val {
                    arch = Some(s.clone());
                }
            }
            "general.name" => {
                if let Val::S(s) = &val {
                    name = Some(s.clone());
                }
            }
            "general.size_label" => {
                if let Val::S(s) = &val {
                    size_label = Some(s.clone());
                }
            }
            "general.file_type" => {
                file_type = as_u32(&val);
            }
            _ => {
                // chaves dependentes da arquitetura: "<arch>.block_count" etc.
                if key.ends_with(".block_count") {
                    block_count = as_u32(&val);
                } else if key.ends_with(".context_length") {
                    context_length = as_u32(&val);
                } else if key.ends_with(".embedding_length") {
                    embedding_length = as_u32(&val);
                } else if key.ends_with(".attention.head_count_kv") {
                    head_count_kv = as_u32(&val);
                } else if key.ends_with(".attention.head_count") {
                    head_count = as_u32(&val);
                }
            }
        }
    }

    Ok(ParsedGguf {
        arch: arch.unwrap_or_else(|| "desconhecida".into()),
        name,
        block_count,
        context_length,
        embedding_length,
        head_count,
        head_count_kv,
        size_label,
        file_type,
    })
}

pub struct ParsedGguf {
    pub arch: String,
    pub name: Option<String>,
    pub block_count: Option<u32>,
    pub context_length: Option<u32>,
    pub embedding_length: Option<u32>,
    pub head_count: Option<u32>,
    pub head_count_kv: Option<u32>,
    pub size_label: Option<String>,
    pub file_type: Option<u32>,
}

// Deriva o rotulo de quantizacao a partir do nome do arquivo (mais confiavel
// que o enum file_type), com fallback para file_type.
fn quant_from_name(fname: &str, file_type: Option<u32>) -> String {
    let upper = fname.to_uppercase();
    const PATTERNS: [&str; 18] = [
        "Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q5_0", "Q5_1", "Q4_K_M", "Q4_K_S", "Q4_0", "Q4_1",
        "Q3_K_L", "Q3_K_M", "Q3_K_S", "Q2_K", "IQ4_XS", "IQ3_XS", "BF16", "F16",
    ];
    for p in PATTERNS {
        if upper.contains(p) {
            return p.to_string();
        }
    }
    match file_type {
        Some(0) => "F32".into(),
        Some(1) => "F16".into(),
        Some(7) => "Q8_0".into(),
        Some(t) => format!("ftype{}", t),
        None => "?".into(),
    }
}

fn is_mmproj(fname: &str) -> bool {
    fname.to_lowercase().starts_with("mmproj")
}

fn collect_gguf(dir: &Path, out: &mut Vec<PathBuf>, depth: usize) {
    if depth > 8 {
        return;
    }
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_dir() {
            collect_gguf(&p, out, depth + 1);
        } else if p.extension().map(|e| e.eq_ignore_ascii_case("gguf")).unwrap_or(false) {
            out.push(p);
        }
    }
}

/// Varre uma lista de diretorios e retorna os modelos GGUF (ignorando os
/// arquivos mmproj-*, que sao anexados ao modelo correspondente).
pub fn scan_dirs(dirs: &[String]) -> Vec<ModelInfo> {
    let mut files: Vec<PathBuf> = Vec::new();
    for d in dirs {
        collect_gguf(Path::new(d), &mut files, 0);
    }

    // mapeia pasta -> mmproj encontrado, para anexar
    let mut models: Vec<ModelInfo> = Vec::new();
    for path in &files {
        let fname = path
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        if is_mmproj(&fname) {
            continue;
        }
        let size_bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        let folder = path
            .parent()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_default();

        // procura um mmproj irmao na mesma pasta
        let mut mmproj_path: Option<String> = None;
        if let Some(parent) = path.parent() {
            if let Ok(siblings) = std::fs::read_dir(parent) {
                for s in siblings.flatten() {
                    let sn = s.file_name().to_string_lossy().into_owned();
                    if is_mmproj(&sn) && sn.to_lowercase().ends_with(".gguf") {
                        mmproj_path = Some(s.path().to_string_lossy().into_owned());
                        break;
                    }
                }
            }
        }

        let parsed = parse_header(path).ok();
        let (
            arch,
            block_count,
            context_length,
            embedding_length,
            head_count,
            head_count_kv,
            size_label,
            gen_name,
            ftype,
        ) = match parsed {
            Some(p) => (
                p.arch,
                p.block_count,
                p.context_length,
                p.embedding_length,
                p.head_count,
                p.head_count_kv,
                p.size_label,
                p.name,
                p.file_type,
            ),
            None => (
                "desconhecida".into(),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ),
        };

        let display_name = gen_name.unwrap_or_else(|| {
            path.file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| fname.clone())
        });

        models.push(ModelInfo {
            path: path.to_string_lossy().into_owned(),
            name: display_name,
            folder,
            arch,
            quant: quant_from_name(&fname, ftype),
            size_bytes,
            size_gb: (size_bytes as f64) / 1e9,
            block_count,
            context_length,
            embedding_length,
            head_count,
            head_count_kv,
            size_label,
            has_mmproj: mmproj_path.is_some(),
            mmproj_path,
        });
    }

    models.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
    models
}
