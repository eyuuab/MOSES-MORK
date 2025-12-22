{-# LANGUAGE ForeignFunctionInterface #-}

module MyLib where

import Foreign.C.Types

-- The actual Haskell logic
haskellAdd :: Int -> Int -> Int
haskellAdd x y = x + y

-- The C-compatible wrapper
-- We convert CInt to Int, do the work, and convert back
foreign export ccall "my_add" my_add :: CInt -> CInt -> CInt

my_add :: CInt -> CInt -> CInt
my_add x y = fromIntegral (haskellAdd (fromIntegral x) (fromIntegral y))