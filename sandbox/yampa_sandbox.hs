{-# LANGUAGE Arrows #-}
import FRP.Yampa
import Control.Concurrent

signalFunction :: SF Double Double
signalFunction = proc x -> do
  y <- integral -< x
  t <- time     -< ()
  returnA -< y / t

firstSample :: IO Double   -- sample at time zero
firstSample =
  return 1.0  -- time == 0, input == 1.0

nextSamples :: Bool -> IO (Double, Maybe Double)
nextSamples x = do
  threadDelay 1000000
  return (0.1, Just 1.0) -- time delta == 0.1s, input == 1.0

output :: Bool -> Double -> IO Bool
output _ x = do
  print x     -- print the output
  return False -- just continue forever

main :: IO ()
main = 
  reactimate firstSample nextSamples output signalFunction

