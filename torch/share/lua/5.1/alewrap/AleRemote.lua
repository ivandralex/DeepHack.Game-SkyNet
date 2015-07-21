--[[ Copyright 2014 Google Inc.

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
]]

function alewrap.createRemote(host, port, romName, extraConfig)
    return alewrap.AleRemoteEnv(host, port, romName, extraConfig)
end

local RAM_LENGTH = 128

-- Copies values from src to dst.
local function update(dst, src)
    for k, v in pairs(src) do
        dst[k] = v
    end
end

-- Copies the config. An error is raised on unknown params.
local function updateDefaults(dst, src)
    for k, v in pairs(src) do
        if dst[k] == nil then
            _print_usage(dst)
            error("unsupported param: " .. k)
        end
    end
    update(dst, src)
end

require 'torch'
local Env = torch.class('alewrap.AleRemoteEnv')
function Env:__init(host, port, romName, extraConfig)
		print("CLIENT 0")
    self.config = {
        -- An additional reward signal can be provided
        -- after the end of one game.
        -- Note that many games don't change the score
        -- when loosing or gaining a life.
        gameOverReward=0,
        -- Screen display can be enabled.
        display=false,
        -- The RAM can be returned as an additional observation.
        enableRamObs=false,
    }
    updateDefaults(self.config, extraConfig)

    self.win = nil
    --self.ale = alewrap.newAle(romPath)
		self.ale_conn = i
		local socket = require("socket")
		self.client = socket.tcp();
		print("CLIENT 1")
		self.client:settimeout(1)
		print("CLIENT 2")
		self.client:connect(host, port);
		print("CLIENT 3")
		msg = self.client:receive('*l')
		self.width = tonumber(string.sub(msg,1,3))
		self.height = tonumber(string.sub(msg,5,7))
		
print("MSG: " , msg, self.width, self.height)
		
		self.client:send('1,1,1,1\n')
		self.last_chunk = self:receiveMain()		

    local obsShapes = {{self.height, self.width}}
    if self.config.enableRamObs then
        obsShapes={{self.height, self.width}, {RAM_LENGTH}}
    end
    self.envSpec = {
        nActions=18,
        obsShapes=obsShapes,
    }
end

function string.fromhex(str)
    return (str:gsub('..', function (cc)
        return string.char(tonumber(cc, 16))
    end))
end

function string.tohex(str)
    return (str:gsub('.', function (c)
        return string.format('%02X', string.byte(c))
    end))
end


function hex_to_byte_tensor_2d(str, w, h)
	t = torch.ByteTensor(h, w)
	s = t:storage()
	i = 1
	for j = 1,str:len(),2 do
    -- do something with c
		s[i] = tonumber(string.sub(str, j, j+1), 16)
		i = i + 1
	end
	return t
end

function hex_to_byte_tensor_1d(str, l)
	t = torch.ByteTensor(l)
	i = 1
	for cc in str:gmatch".." do
    -- do something with c
		t[i] = tonumber(cc, 16)
		i = 1 + 1
	end
	return t
end


function Env:receiveMain()
	data = self.client:receive('*l')
	--print("RAW:", data)
	ram_string = string.sub(data, 1, 256)
	--print("RAM:", ram_string)
	ram_obs = hex_to_byte_tensor_1d(ram_string, 128)
	screen_sz =  self.width * self.height * 2
	screen_string = string.sub(data, 258, 257 + screen_sz)
	--print("SCREEN:", screen_string)
	screen_obs = hex_to_byte_tensor_2d(screen_string, self.width, self.height)
	episode_string = string.sub(data, 256 + 1 + screen_sz + 1)

	--print("EPISODE: ", episode_string)	
	terminal, reward = string.match(episode_string, "(%d+),(%d+)")
	--print(ram_obs:size())
	--print(screen_obs:size())
	chunk = { ram_obs, screen_obs, tonumber(terminal), tonumber(reward) }
	--print(chunk[3],chunk[4])
	return chunk	
end

function Env:isGameOver()
	return self.last_chunk[3] == 1 
end

function Env:play(action)
	msg = tostring(action) .. ",18\n"
	--print("SEND: " .. msg) 
	self.client:send(msg)
	self.last_chunk = self:receiveMain()		
end

-- Returns a description of the observation shapes
-- and of the possible actions.
function Env:getEnvSpec()
    return self.envSpec
end

-- Returns a list of observations.
-- The integer palette values are returned as the observation.
function Env:envStart()
		--self.ale:resetGame()
    return self:_generateObservations()
end

-- Does the specified actions and returns the (reward, observations) pair.
-- Valid actions:
--     {torch.Tensor(zeroBasedAction)}
-- The action number should be an integer from 0 to 17.
function Env:envStep(actions)
    assert(#actions == 1, "one action is expected")
    assert(actions[1]:nElement() == 1, "one discrete action is expected")

		--print(" Env:envStep(", actions, ")", self.last_chunk[3], self.last_chunk[4])
    if self:isGameOver() then
        --self.ale:resetGame()
				self:play(45)
        -- The first screen of the game will be also
        -- provided as the observation.
        return self.config.gameOverReward, self:_generateObservations()
    end

		self:play(actions[1][1])
		return self.last_chunk[4], self:_generateObservations()
end

function Env:getRgbFromPalette(obs)
    return alewrap.getRgbFromPalette(obs)
end

function Env:_createObs()
    -- The torch.data() function is provided by torchffi.
    --local obs = torch.ByteTensor(self.height, self.width)
    --self.ale:fillObs(torch.data(obs), obs:nElement())
    --return obs
		return self.last_chunk[2]
end

function Env:_createRamObs()
    --local ram = torch.ByteTensor(RAM_LENGTH)
    --self.ale:fillRamObs(torch.data(ram), ram:nElement())
    --return ram
    return self.last_chunk[1]
end

function Env:_display(obs)
    require 'image'
    local frame = self:getRgbFromPalette(obs)
    self.win = image.display({image=frame, win=self.win})
end

-- Generates the observations for the current step.
function Env:_generateObservations()
    local obs = self:_createObs()
    if self.config.display then
        self:_display(obs)
    end

    if self.config.enableRamObs then
        local ram = self:_createRamObs()
        return {obs, ram}
    else
        return {obs}
    end
end

function Env:saveState()
    --self.ale:saveState()
		self:play(43)
end

function Env:loadState()
    --return self.ale:loadState()
		self:play(44)
end

function Env:actions()
    --local nactions = self.ale:numActions()
    --local actions = torch.IntTensor(nactions)
    --self.ale:actions(torch.data(actions), actions:nElement())
    local actions = torch.IntTensor(18)
		for i = 1,18 do
			actions[i] = i - 1
		end
    return actions
end

function Env:lives()
		return nil
    --return self.ale:lives()
end

function Env:saveSnapshot()
    return self.ale:saveSnapshot()
end

function Env:restoreSnapshot(snapshot)
    self.ale:restoreSnapshot(snapshot)
end

function Env:getScreenWidth()
  return self.width
end

function Env:getScreenHeight()
  return self.height
end

