#[unsafe(no_mangle)]
pub fn unsigned_checked(bytes:&[u8])->(u64,&[u8]) {unsigned_varint::decode::u64(bytes).expect("internal encoded bytes")}
#[unsafe(no_mangle)]
pub unsafe fn unsigned_trusted(bytes:&[u8])->(u64,&[u8]) {unsafe{unsigned_varint::decode::u64(bytes).unwrap_unchecked()}}
#[unsafe(no_mangle)]
pub fn vint_checked(bytes:&mut &[u8])->u64 {vint64::decode(bytes).expect("internal encoded bytes")}
#[unsafe(no_mangle)]
pub unsafe fn vint_trusted(bytes:&mut &[u8])->u64 {unsafe{vint64::decode(bytes).unwrap_unchecked()}}
