{-# LANGUAGE ForeignFunctionInterface #-}

module MyLib where

import Foreign.C.Types
import Data.List (intercalate, (\\))

-- 1. Your Core Logic
class Quantale q where
  qjoin     :: q -> q -> q
  qtimes    :: q -> q -> q
  qresiduum :: q -> q -> q
  qunit     :: q        

newtype LogicalQuantale = Logical Bool deriving (Show, Eq)

instance Quantale LogicalQuantale where
    qjoin (Logical a) (Logical b) = Logical (a || b)
    qtimes (Logical a) (Logical b) = Logical (a && b)
    qresiduum (Logical a) (Logical b) = Logical (not a || b)
    qunit = Logical True

-- 2. The FFI Wrappers (Monomorphic)
-- We map LogicalQuantale (Bool) to CBool (usually an 8-bit int in C)

foreign export ccall "q_join_logical" q_join_logical :: CBool -> CBool -> CBool
q_join_logical a b = 
    let (Logical res) = qjoin (Logical (toBool a)) (Logical (toBool b))
    in fromBool res

foreign export ccall "q_times_logical" q_times_logical :: CBool -> CBool -> CBool
q_times_logical a b = 
    let (Logical res) = qtimes (Logical (toBool a)) (Logical (toBool b))
    in fromBool res

-- Helper functions to convert between C and Haskell booleans
toBool :: CBool -> Bool
toBool 0 = False
toBool _ = True

fromBool :: Bool -> CBool
fromBool True  = 1
fromBool False = 0

-- Keep your initialization functions in the C stub as before