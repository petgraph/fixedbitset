#[cfg(not(feature = "std"))]
use core as std;

use std::{convert::TryFrom, fmt};
use crate::{FixedBitSet, BYTES};
use serde::de::{self, Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use serde::ser::{Serialize, SerializeStruct, Serializer};

struct BitSetByteSerializer<'a>(&'a FixedBitSet);

impl Serialize for FixedBitSet {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut struct_serializer = serializer.serialize_struct("FixedBitset", 2)?;
        struct_serializer.serialize_field("length", &(self.length as u64))?;
        struct_serializer.serialize_field("data", &BitSetByteSerializer(self))?;
        struct_serializer.end()
    }
}

impl<'a> Serialize for BitSetByteSerializer<'a> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let len = self.0.data.len() * BYTES;
        // PERF: Figure out a way to do this without allocating.
        let mut temp = Vec::with_capacity(len);
        for block in &self.0.data {
            temp.extend(&block.to_le_bytes());
        }
        serializer.serialize_bytes(&temp)
    }
}

impl<'de> Deserialize<'de> for FixedBitSet {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        enum Field {
            Length,
            Data,
        }

        impl<'de> Deserialize<'de> for Field {
            fn deserialize<D>(deserializer: D) -> Result<Field, D::Error>
            where
                D: Deserializer<'de>,
            {
                struct FieldVisitor;

                impl<'de> Visitor<'de> for FieldVisitor {
                    type Value = Field;

                    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                        formatter.write_str("`length` or `data`")
                    }

                    fn visit_str<E>(self, value: &str) -> Result<Field, E>
                    where
                        E: de::Error,
                    {
                        match value {
                            "length" => Ok(Field::Length),
                            "data" => Ok(Field::Data),
                            _ => Err(de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }

                deserializer.deserialize_identifier(FieldVisitor)
            }
        }

        struct FixedBitSetVisitor;

        impl<'de> Visitor<'de> for FixedBitSetVisitor {
            type Value = FixedBitSet;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct Duration")
            }

            fn visit_seq<V>(self, mut seq: V) -> Result<FixedBitSet, V::Error>
            where
                V: SeqAccess<'de>,
            {
                let length = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(0, &self))?;
                let data = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(1, &self))?;
                Ok(FixedBitSet { length, data })
            }

            fn visit_map<V>(self, mut map: V) -> Result<FixedBitSet, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut length = None;
                let mut temp: Option<&[u8]> = None;
                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Length => {
                            if length.is_some() {
                                return Err(de::Error::duplicate_field("length"));
                            }
                            length = Some(map.next_value()?);
                        }
                        Field::Data => {
                            if temp.is_some() {
                                return Err(de::Error::duplicate_field("data"));
                            }
                            temp = Some(map.next_value()?);
                        }
                    }
                }
                let length = length.ok_or_else(|| de::Error::missing_field("length"))?;
                let temp = temp.ok_or_else(|| de::Error::missing_field("data"))?;
                let block_len = length / BYTES + 1;
                let mut data = Vec::with_capacity(block_len);
                for chunk in temp.chunks(BYTES) {
                    match <&[u8; BYTES]>::try_from(chunk) {
                        Ok(bytes) => data.push(usize::from_le_bytes(*bytes)),
                        Err(_) => {
                            let mut bytes = [0u8; BYTES];
                            bytes[0..BYTES].copy_from_slice(chunk);
                            data.push(usize::from_le_bytes(bytes));
                        }
                    }
                }
                Ok(FixedBitSet { length, data })
            }
        }

        const FIELDS: &'static [&'static str] = &["length", "data"];
        deserializer.deserialize_struct("Duration", FIELDS, FixedBitSetVisitor)
    }
}
