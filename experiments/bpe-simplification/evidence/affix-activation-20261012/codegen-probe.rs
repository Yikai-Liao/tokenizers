#[derive(Clone, Copy)]
#[repr(usize)]
enum Affix { Prefix=0, Suffix=1, Both=2, Plain=3 }
#[unsafe(no_mangle)]
pub fn tuple_match(p:bool,s:bool,first:bool,last:bool)->usize {
 let a=match(p&&!first,s&&last) {(false,false)=>Affix::Plain,(true,false)=>Affix::Prefix,(false,true)=>Affix::Suffix,(true,true)=>Affix::Both};a as usize
}
#[unsafe(no_mangle)]
pub fn two_choices(p:bool,s:bool,first:bool,last:bool)->usize {
 let prefix=p&&!first;
 let plain_or_prefix=if prefix { Affix::Prefix } else { Affix::Plain };
 let suffix_or_both=if prefix { Affix::Both } else { Affix::Suffix };
 let a=if s&&last {suffix_or_both} else {plain_or_prefix};a as usize
}
#[unsafe(no_mangle)]
pub fn nested_choice(p:bool,s:bool,first:bool,last:bool)->usize {
 let prefix=p&&!first;
 let a=if s&&last { if prefix {Affix::Both} else {Affix::Suffix} } else if prefix {Affix::Prefix} else {Affix::Plain};a as usize
}
#[unsafe(no_mangle)]
pub fn staged_choice(p:bool,s:bool,first:bool,last:bool)->usize {
 let a=if p&&!first {Affix::Prefix} else {Affix::Plain};
 let a=if s&&last {if matches!(a,Affix::Prefix) {Affix::Both}else {Affix::Suffix}} else {a};a as usize
}
#[unsafe(no_mangle)]
pub fn paired_choice(p:bool,s:bool,first:bool,last:bool)->usize {
 let (without,with)=if p&&!first {(Affix::Prefix,Affix::Both)} else {(Affix::Plain,Affix::Suffix)};
 let a=if s&&last {with}else{without};a as usize
}
#[unsafe(no_mangle)]
pub fn equal_choice(p:bool,s:bool,first:bool,last:bool)->usize {
 let p=p&&!first;let s=s&&last;
 let a=if p==s {if p {Affix::Both} else{Affix::Plain}} else if p {Affix::Prefix} else {Affix::Suffix};a as usize
}
#[unsafe(no_mangle)]
pub fn paired_booland(p:bool,s:bool,first:bool,last:bool)->usize {
 let (without,with)=if p & !first {(Affix::Prefix,Affix::Both)} else {(Affix::Plain,Affix::Suffix)};
 let a=if s & last {with}else{without};a as usize
}
