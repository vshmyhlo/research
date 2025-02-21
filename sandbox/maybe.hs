import Control.Applicative
import Control.Monad (MonadPlus (mzero), guard)
import Data.Foldable

-- import Text.Read (Lexeme(Number))
-- {-# LANGUAGE ScopedTypeVariables #-}
-- import Data.Maybe

lst :: [Int]
lst = [1, 2, 3]

-- x :: Int
-- x = mzero

-- f :: Int -> Int
-- f 1 = 1
-- f 2 =2

sandbox :: Maybe Int
sandbox = do
  0 <- Just 0
  a <- Just 1
  b <- Just 2
  guard (a == 1)
  let Just c = Just 1
  return (a + b + c)

main = print $ show sandbox
