import Control.Monad.Cont

ex2 = do
  c1 <- cont (\c -> c "cont-1.1" ++ " and " ++ c "cont-1.2")
  c2 <- cont (\c -> c "cont-2")
  return $ "c1=" ++ c1 ++ ", c2=" ++ c2

-- return $ a ++ " " ++ b

test2 = runCont ex2 (\x -> "(" ++ x ++ ")")

main = print test2
