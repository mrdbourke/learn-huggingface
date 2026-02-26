-- filters/cell-numbers.lua
-- Preserves real Jupyter execution_count as In[N]/Out[N] labels

function Div(el)
  if not el.classes:includes("cell") then
    return nil
  end

  local exec_count = el.attributes["execution_count"]
  if exec_count == nil then
    return nil
  end

  local new_content = {}
  local out_label_inserted = false

  -- In [N]: label goes at the very start (code is always first)
  table.insert(new_content, pandoc.RawBlock("html",
    '<div class="cell-label cell-label-in">In [' .. exec_count .. ']:</div>'
  ))

  for _, block in ipairs(el.content) do
    -- Out [N]: label before the first output block
    if block.t == "Div" and not out_label_inserted then
      for _, cls in ipairs(block.classes) do
        if cls:match("^cell%-output") then
          table.insert(new_content, pandoc.RawBlock("html",
            '<div class="cell-label cell-label-out">Out [' .. exec_count .. ']:</div>'
          ))
          out_label_inserted = true
          break
        end
      end
    end
    table.insert(new_content, block)
  end

  el.content = new_content
  return el
end