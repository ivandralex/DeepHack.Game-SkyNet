require("math")

local meta = {}
function meta:__index(k) return k end
function PositiveIntegers() return setmetatable({}, meta) end
  
local function sum(table)
    local sum = 0
    for k, v in pairs(table) do
        sum = sum + v
    end 
    return sum
end 

local function permute(tab, n, count)
	n = n or #tab
  		for i = 1, count or n do
    	local j = math.random(i, n)
    	tab[i], tab[j] = tab[j], tab[i]
  	end
  	return tab
end

function lotto(count, range)
  	return {unpack(permute(PositiveIntegers(), range, count), 1, count)}
end